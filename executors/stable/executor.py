import PIL
import importlib
import numpy as np
import os
import shutil
import sys
import time
import torch
import torch.nn as nn

from PIL import Image
from contextlib import nullcontext
from einops import rearrange, repeat
from io import BytesIO
from itertools import islice
from pathlib import Path
from pytorch_lightning import seed_everything
from random import randint
from torch import autocast
from tqdm import tqdm, trange
from typing import Dict

from jina import Executor, DocumentArray, Document, requests
from omegaconf import OmegaConf

VALID_SAMPLERS = {'ddim'}

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optimizedSD


class StableDiffusionConfig:
    '''
    Configuration for Stable Diffusion.
    '''
    C = 4 # latent channels
    ckpt = '' # model checkpoint path
    config = '' # model configuration file path
    ddim_eta = 0.0
    ddim_steps = 50
    f = 8 # downsampling factor
    fixed_code = False
    height = 512
    n_iter = 1 # number of times to sample
    n_samples = 1 # batch size, GPU memory use scales quadratically with this but it makes it sample faster!
    precision = 'autocast'
    scale = 7.5 # unconditional guidance scale
    seed = 1
    unet_bs = 1
    width = 512


class KCFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    return sd


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


class StableDiffusionGenerator(Executor):
    '''
    Executor generator for all stable diffusion API paths.
    '''
    opt: StableDiffusionConfig = StableDiffusionConfig()

    config = ''
    device = None
    input_path = ''
    model = None
    modelCS = None
    modelFS = None
    model_k_wrapped = None
    model_k_config = None
    sample = None

    def __init__(self,
        stable_path: str,
        height: int=512,
        n_iter: int=1,
        n_samples: int=4,
        width: int=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_path = stable_path
        self.opt.config = f'{stable_path}/configs/stable-diffusion/v1-inference.yaml'
        self.opt.ckpt = f'{stable_path}/models/ldm/stable-diffusion-v1/model.ckpt'

        self.opt.height = height
        self.opt.width = width
        self.opt.n_samples = n_samples
        self.opt.n_iter = n_iter

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.config = OmegaConf.load(Path(__file__).resolve().parent / 'v1-inference.yml')

        sd = load_model_from_config(f"{self.opt.ckpt}")
        li, lo = [], []
        for key, value in sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd["model1." + key[6:]] = sd.pop(key)
        for key in lo:
            sd["model2." + key[6:]] = sd.pop(key)

        model = instantiate_from_config(self.config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.eval()
        model.unet_bs = self.opt.unet_bs
        model.cdevice = self.device

        modelCS = instantiate_from_config(self.config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.eval()
        modelCS.cond_stage_model.device = self.device

        modelFS = instantiate_from_config(self.config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()
        del sd

        model.half()
        modelCS.half()

        self.model = model
        self.modelCS = modelCS
        self.modelFS = modelFS

        self.sample = model.sample

        model.make_schedule(
            ddim_num_steps=self.opt.ddim_steps, ddim_eta=self.opt.ddim_eta,
                verbose=False)

    @requests(on='/')
    def txt2img(self, docs: DocumentArray, parameters: Dict, **kwargs):
        request_time = time.time()

        sampler = parameters.get('sampler', 'ddim')
        if sampler not in VALID_SAMPLERS:
            raise ValueError(f'sampler must be in {VALID_SAMPLERS}, got {sampler}')
        scale = parameters.get('scale', 7.5)
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        opt = self.opt
        opt.scale = scale
        steps = min(int(parameters.get('steps', opt.ddim_steps)), 250)

        # If the number of samples we have is more than would currently be
        # given for n_samples * n_iter, increase n_iter to yield more images.
        n_samples = opt.n_samples
        n_iter = opt.n_iter
        if num_images < n_samples:
            n_samples = num_images
        if num_images // n_samples > n_iter:
            n_iter = num_images // n_samples
        seed_everything(seed)

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([n_samples, opt.C, opt.height // opt.f,
                opt.width // opt.f], device=self.device)

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                for d in docs:
                    batch_size = n_samples
                    prompt = d.text
                    assert prompt is not None
                    data = [batch_size * [prompt]]

                    self.logger.info(f'stable diffusion start {num_images} images, prompt "{prompt}"...')
                    for n in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            self.modelCS.to(self.device)
                            uc = None
                            if opt.scale != 1.0:
                                uc = self.modelCS.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.modelCS.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.height // opt.f, opt.width // opt.f]

                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                            torch.cuda.empty_cache()

                            self.modelFS.to(self.device)  

                            samples = None
                            if sampler == 'ddim':
                                samples = self.sample(
                                    seed=seed,
                                    S=steps,
                                    conditioning=c,
                                    batch_size=n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=start_code)

                            x_samples_ddim = self.modelFS.decode_first_stage(samples)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                buffered = BytesIO()
                                img.save(buffered, format='PNG')
                                _d = Document(
                                    blob=buffered.getvalue(),
                                    mime_type='image/png',
                                    tags={
                                        'text': prompt,
                                        'generator': 'stable-diffusion',
                                        'request_time': request_time,
                                        'created_time': time.time(),
                                    },
                                ).convert_blob_to_datauri()
                                _d.text = prompt
                                d.matches.append(_d)
                            
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                            torch.cuda.empty_cache()
    

    @requests(on='/stablediffuse')
    def stablediffuse(self, docs: DocumentArray, parameters: Dict, **kwargs):
        '''
        Called "img2img" in the scripts of the stable-diffusion repo.
        '''
        request_time = time.time()

        latentless = parameters.get('latentless', False)
        num_images = max(1, min(8, int(parameters.get('num_images', 1))))
        prompt_override = parameters.get('prompt', None)
        sampler = parameters.get('sampler', 'ddim')
        scale = parameters.get('scale', 7.5)
        seed = int(parameters.get('seed', randint(0, 2 ** 32 - 1)))
        strength = parameters.get('strength', 0.75)

        if sampler not in VALID_SAMPLERS:
            raise ValueError(f'sampler must be in {VALID_SAMPLERS}, got {sampler}')

        opt = self.opt
        opt.scale = scale

        # If the number of samples we have is more than would currently be
        # given for n_samples * n_iter, increase n_iter to yield more images.
        n_samples = opt.n_samples
        n_iter = opt.n_iter
        if num_images < n_samples:
            n_samples = num_images
        if num_images // n_samples > n_iter:
            n_iter = num_images // n_samples
        
        seed_everything(seed)

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * opt.ddim_steps)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                for d in docs:
                    batch_size = n_samples
                    prompt = d.text
                    if prompt_override is not None:
                        prompt = prompt_override
                    assert prompt is not None
                    self.logger.info(f'stable diffusion img2img start {num_images} images, prompt "{prompt}"...')
                    data = [batch_size * [prompt]]

                    input_path = os.path.join(self.input_path, f'{d.id}/')

                    Path(input_path).mkdir(parents=True, exist_ok=True)
                    Path(os.path.join(input_path, 'out')).mkdir(parents=True, exist_ok=True)

                    temp_file_path = os.path.join(input_path, f'{d.id}.png')
                    d.save_uri_to_file(temp_file_path)

                    assert os.path.isfile(temp_file_path)
                    init_image = load_img(temp_file_path).to(self.device)
                    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

                    self.modelFS.to(self.device)

                    init_latent = None
                    if not latentless:
                        init_latent = self.modelFS.get_first_stage_encoding(
                            self.modelFS.encode_first_stage(init_image))  # move to latent space
                    else:
                        init_latent = torch.zeros(
                            batch_size,
                            4,
                            opt.height >> 3,
                            opt.width >> 3,
                        ).cuda()

                    mem = torch.cuda.memory_allocated() / 1e6
                    self.modelFS.to("cpu")
                    while torch.cuda.memory_allocated() / 1e6 >= mem:
                        time.sleep(1)
                    torch.cuda.empty_cache()

                    for n in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            self.modelCS.to(self.device)

                            uc = None
                            if opt.scale != 1.0:
                                uc = self.modelCS.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.modelCS.get_learned_conditioning(prompts)

                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                            torch.cuda.empty_cache()

                            samples = None
                            # encode (scaled latent)
                            z_enc = self.model.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc]*batch_size).to(self.device),
                                seed,
                                opt.ddim_eta,
                                opt.ddim_steps,
                            )
                            # decode it
                            samples = self.model.decode(z_enc, c, t_enc,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc)

                            self.modelFS.to(self.device)

                            x_samples = self.modelFS.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                buffered = BytesIO()
                                img.save(buffered, format='PNG')
                                _d = Document(
                                    blob=buffered.getvalue(),
                                    mime_type='image/png',
                                    tags={
                                        'text': prompt,
                                        'generator': 'stable-diffusion',
                                        'request_time': request_time,
                                        'created_time': time.time(),
                                    },
                                ).convert_blob_to_datauri()
                                _d.text = prompt
                                d.matches.append(_d)

                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                            torch.cuda.empty_cache()

                        shutil.rmtree(input_path, ignore_errors=True)
