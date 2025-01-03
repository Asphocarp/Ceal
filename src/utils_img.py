# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyright: reportMissingModuleSource=false

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# me: vqgan is in range [-1,1], imnet is in range [-2.2,2.2]
#     common pic is in range [0,1]

normalize_vqgan = transforms.Normalize(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(
    mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])  # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])  # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[
                                       1/0.229, 1/0.224, 1/0.225])  # Unnormalize (x * std) + mean

normalize_yuv = transforms.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1])
unnormalize_yuv = transforms.Normalize(
    mean=[-0.5/0.5, 0, 0], std=[1/0.5, 1/1, 1/1])


def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - \
            torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - \
            torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20*np.log10(255) - 10 * \
        torch.log10(torch.mean(delta**2, dim=(1, 2, 3)))  # B
    return psnr


def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)


def resized_center_crop(x, scale):
    '''Remain original size'''
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    top, left = (x.shape[-2] - new_edges_size[0]) // 2, (x.shape[-1] - new_edges_size[1]) // 2
    return functional.resized_crop(x, top, left, new_edges_size[0], new_edges_size[1], x.shape[-2:])


def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)


def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)


def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))


def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))


def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))


def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))


def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))


def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))


def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(
            aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)


image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def normalize_img(x):
    """ Normalize image to approx. [-1,1] """
    return (x - image_mean.to(x.device)) / image_std.to(x.device)

def unnormalize_img(x):
    """ Unnormalize image to [0,1] """
    return (x * image_std.to(x.device)) + image_mean.to(x.device)

def round_pixel(x):
    """ 
    Round pixel values to nearest integer. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y


def clamp_pixel(x):
    """ 
    Clamp pixel values to 0 255. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = x_pixel.clamp(0, 255)
    y = normalize_img(y/255.0)
    return y


def project_linf(x, y, radius):
    """ 
    Clamp x so that Linf(x,y)<=radius
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        radius: Radius of Linf ball for the images in pixel space [0, 255]
     """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std.to(x.device)
    return y + delta


def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20*np.log10(255) - 10 * \
        torch.log10(torch.mean(delta**2, dim=(1, 2, 3)))  # B
    return psnr

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = unnormalize_img(x)
    for ii, img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(
            pil_img, quality=quality_factor))
    return normalize_img(img_aug)


def gaussian_blur(x, sigma=1):
    """ Add gaussian blur to image
    Args:
        x: Tensor image
        sigma: sigma of gaussian kernel
    """
    x = unnormalize_img(x)
    x = functional.gaussian_blur(x, sigma=sigma, kernel_size=21)
    x = normalize_img(x)
    return x
