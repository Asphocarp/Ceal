# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import time
import datetime
import os
import subprocess
import functools
from collections import defaultdict, deque
from typing import *

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torchvision.datasets.folder import is_image_file, default_loader
from torch.optim import Optimizer
import os
import pandas as pd
from torchvision.io import read_image
from skimage import io
from pathlib import Path


class ImageDatasetWithCaption(Dataset):
    def __init__(self, 
                 path,
                 annotations_file,
                 transform=None,
                 loader=default_loader,
    ):
        from pycocotools.coco import COCO
        # self.img_labels = pd.read_csv(annotations_file)
        self.coco_caps = COCO(annotations_file)
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img_path = self.samples[idx]
        img = self.loader(img_path)
        if self.transform:
            img = self.transform(img)
        # get label
        stem = Path(img_path).stem
        imgId = int(stem[-7:])
        annId = self.coco_caps.getAnnIds(imgId)[0]
        caption = self.coco_caps.loadAnns(annId)[0]['caption']
        # caption
        return img, caption, imgId

def attr(obj, attr):
    # return None if non-existent
    return getattr(obj, attr, None)


### Optimizer building

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]] = float(x[1])
    return params


def build_optimizer(name, model_params, **optim_params) -> Optimizer:
    """ Build optimizer from a dictionary of parameters """
    torch_optimizers = sorted(name for name in torch.optim.__dict__
                              if name[0].isupper() and not name.startswith("__")
                              and callable(torch.optim.__dict__[name]))
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(torch_optimizers)}')


def adjust_learning_rate(optimizer, step, steps, warmup_steps, blr, min_lr=1e-6,
                         cosine_lr=True):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < warmup_steps:
        lr = blr * step / warmup_steps
    elif not cosine_lr:
        lr = blr
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_lambda_i(step, max_steps, max_i=0.2, min_i=1e-3):
    """
    Increase the lambda_i with half-cycle cosine after warmup
        max_steps: after which become max
    """
    if step >= max_steps:
        ret = max_i
    else:
        ret = max_i + (min_i - max_i) * 0.5 * \
              (1. + math.cos(math.pi * step / max_steps))
    return ret


### Data loading

@functools.lru_cache()
def get_image_paths(path):
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch


def get_dataloader(
    data_dir, transform, 
    batch_size=128, num_imgs=None, 
    shuffle=False, num_workers=4,
    collate_fn=collate_fn
):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if num_imgs is not None:
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      drop_last=False, collate_fn=collate_fn)

def get_dataloader_with_caption(
    data_dir, transform, 
    batch_size=128, num_imgs=None, 
    shuffle=False, num_workers=4,
    collate_fn=collate_fn,
    annotations_file = '../../cache/annotations/captions_val2014.json',
):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageDatasetWithCaption(data_dir, transform=transform,
                                      annotations_file=annotations_file)
    if num_imgs is not None:
        dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      drop_last=False, collate_fn=collate_fn)


def get_dataloader_then_repeat(
    data_dir, transform, 
    batch_size=128, num_imgs=None, 
    shuffle=False, num_workers=4,
    collate_fn=collate_fn
):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if num_imgs is not None:
        if num_imgs < len(dataset):
            dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
        else:
            indices = np.random.choice(len(dataset), len(dataset), replace=False)
            repeat_times = math.ceil(num_imgs/len(dataset))
            indices = np.tile(indices, repeat_times)[:num_imgs]
            dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      drop_last=False, collate_fn=collate_fn)


def get_dataloader_for_folder(data_dir, transform, batch_size=128, num_imgs=None, shuffle=False, num_workers=4, collate_fn=collate_fn):
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)


def latents_dataloader_repeat(filename, batch_size=4, num_imgs=None, num_workers=4, collate_fn=collate_fn):
    latents: List[torch.Tensor] = torch.load(filename)
    elems = []
    for z in latents:
        elems += [e for e in z]
    elems = torch.stack(elems)
    dataset = TensorDataset(elems)
    if num_imgs is not None:
        if num_imgs < len(dataset):
            dataset = Subset(dataset, np.random.choice(len(dataset), num_imgs, replace=False))
        else:
            indices = np.random.choice(len(dataset), len(dataset), replace=False)
            repeat_times = math.ceil(num_imgs/len(dataset))
            indices = np.tile(indices, repeat_times)[:num_imgs]
            dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)


def pil_imgs_from_folder(folder):
    """ Get all images in the folder as PIL images """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                filenames.append(filename)
                images.append(img)
        except:
            print("Error opening image: ", filename)
    return images, filenames


### Metric logging

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / (len(iterable) + 1)))


### Misc

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


### NEW

def seed_everything(seed):
    # Set seeds for reproductibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# find diff in tensor and merge to 2 dim or below
def diffTensor(a: torch.Tensor, b: torch.Tensor, threshold=0, result_dimension=2):
    assert len(a.shape) == len(b.shape)
    same = torch.abs_(a - b) <= threshold
    for _ in range(len(a.shape) - result_dimension):
        same = same.all(dim=-1)
    return same == False


# make torch.nonzero readable
def readableNonZero(x):
    ret = torch.nonzero(x)
    if 0 in ret.size():
        return None
    if len(ret.shape) == 2:
        if ret.shape[-1] == 1:
            ret = ret.squeeze(dim=-1)
            if (ret == torch.arange(ret[0], ret[-1] + 1, device=ret.device)).all():
                ret = f'{ret[0]}~{ret[-1] + 1}'
        elif ret.shape[-1] == 2:
            # to binary image (white for diff)
            # dim: 0 for out, 1 for in (y, x)
            b = (x != 0).cpu().to(dtype=torch.uint8)
            b = (b * 255).numpy().astype(np.uint8)
            img = Image.fromarray(b)
            return img
    return ret


# diff the params of two models
def diffWB(A, B, threshold=0):
    # TODO color for all zero
    ret = {}
    for (an, at), (bn, bt) in zip(list(A.named_parameters()), list(B.named_parameters())):
        name = f'the-{an}' if an == bn else f'{an}&{bn}'
        val = readableNonZero(diffTensor(at, bt, threshold=threshold))
        ret[name] = val
    return ret

def latents_filename(model_id, size=256):
    model = model_id.split('/')[-1]
    filename = f'pretrained/latents_{model}_{size}.pt'
    return filename

def get_mappers(model: nn.Module):
    # -> List[Tuple[str, Mapper]]
    from mapper import Mapper
    ret = []
    for name, module in model.named_modules():
        if isinstance(module, Mapper):
            ret.append((name, module))
    # sort (move mid just after conv_in)
    mid_part = [i for i in ret if 'mid_block' in i[0]]
    not_mid_part = [i for i in ret if 'mid_block' not in i[0]]
    return [not_mid_part[0]] + mid_part + not_mid_part[1:] if not_mid_part else mid_part

def get_convs(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    ret = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # note: ConvWithMapper is a subclass of nn.Conv2d
            ret.append((name, module))
    # sort (move mid just after conv_in)
    mid_part = [i for i in ret if 'mid_block' in i[0]]
    not_mid_part = [i for i in ret if 'mid_block' not in i[0]]
    return [not_mid_part[0]] + mid_part + not_mid_part[1:] if not_mid_part else mid_part

def get_pipe(conf: Config):
    from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, \
        DPMSolverMultistepScheduler, StableDiffusionPipeline, DiTPipeline
    kwargs_from_pretrained = {
        "local_files_only": conf.local_files_only,
        "use_safetensors": conf.use_safetensors,
        "torch_dtype": conf.get_torch_dtype(),
        "variant": "fp16" if conf.torch_dtype_str == "float16" else None,
        "add_watermarker": False,
    }
    if 'sdxl-turbo' in conf.model_id:
        pipe = DiffusionPipeline.from_pretrained(
            conf.model_id,
            **kwargs_from_pretrained,
        )
    elif 'lcm-sdxl' in conf.model_id:
        unet = UNet2DConditionModel.from_pretrained(
            # "latent-consistency/lcm-sdxl",
            # "../../cache/lcm-sdxl",
            conf.model_id,
            **kwargs_from_pretrained,
        )
        pipe = DiffusionPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0", 
            "../../cache/stable-diffusion-xl-base-1.0",
            unet=unet,
            **kwargs_from_pretrained,
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif 'stable-diffusion-xl-base-1.0' in conf.model_id:
        pipe = DiffusionPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0", 
            # "../../cache/stable-diffusion-xl-base-1.0",
            conf.model_id,
            **kwargs_from_pretrained,
        )
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    elif 'stable-diffusion-2-1' in conf.model_id:
        pipe = StableDiffusionPipeline.from_pretrained(
            conf.model_id, 
            **kwargs_from_pretrained,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif 'DiT-XL-2-512' in conf.model_id:
        pipe = DiTPipeline.from_pretrained(
            conf.model_id, 
            **kwargs_from_pretrained,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise NotImplemented
    return pipe

def get_pipe_step_args(conf):
    # conf: Config
    # from config import Config
    if 'sdxl-turbo' in conf.model_id:
        # num_inference_steps=4, guidance_scale=0.0
        # height=512, width=512,
        args = {
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 512,
            "width": 512,
        }
    elif 'lcm-sdxl' in conf.model_id:
        # num_inference_steps=4, guidance_scale=1.0
        args = {
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "height": 768,
            "width": 768,
        }
    elif 'stable-diffusion-xl-base-1.0' in conf.model_id:
        args = {
            "height": 768,
            "width": 768,
        }
    elif 'stable-diffusion-2-1' in conf.model_id:
        args = {
            "height": 512,
            "width": 512,
        }
    elif 'DiT-XL-2-512' in conf.model_id:
        args = {
            "num_inference_steps": 25,
        }
    else:
        raise NotImplemented
    return args