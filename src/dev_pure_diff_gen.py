#!/usr/bin/env python
# %%
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from typing import *
import utils_img
import utils
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
import torch
from copy import deepcopy
from loss.loss_provider import LossProvider
import lpips
from diffusers import StableDiffusionXLPipeline
from diffusers.models.vae import Decoder
from diffusers.models.autoencoder_kl import AutoencoderKL
from pathfinder import Pathfinder, Policy
import argparse
import wandb
from torch import nn
import numpy as np
from tqdm import tqdm
from mapper import MappingNetwork
from PIL import Image, ImageChops
import gc
import misc
import plotly.express as px

#%% ====config====
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default='stabilityai/sdxl-turbo')
args, unknown = parser.parse_known_args()
#>> rename
MODEL_ID = args.model_id
#>> handler
device = 'cuda'

#%% ====main====
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    use_safetensors=True,
    local_files_only=True,
    add_watermark=False,
)
pipe = pipe.to(device)

def gen(
    prompt: str,
    seed: int,
    resolution: int,
    guidance_scale: float = 0.0,
    num_inference_steps: int = 1,
):
    utils.seed_everything(seed)
    z = pipe(prompt=prompt, 
             guidance_scale=guidance_scale, 
             num_inference_steps=num_inference_steps,
             height=resolution, width=resolution,
             output_type='latent')[0]
    z = z / pipe.vae.config.scaling_factor
    z = pipe.vae.post_quant_conv(z)
    ori_norm = pipe.vae.decoder(z)
    ori_nograd = ori_norm.detach().cpu()
    ori_pil = pipe.image_processor.postprocess(ori_nograd, output_type='pil')[0]
    return ori_pil, ori_norm, z

# %%
pil, ori_norm, z = gen('cat', 0, 512)
# show the image using plotly
fig = px.imshow(pil)
fig.show()

# %%
