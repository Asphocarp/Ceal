# %%
import os
import sys
import gc
from typing import *
import utils_img
import utils
from torchvision import transforms
import torch
from copy import deepcopy
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.autoencoder_kl import AutoencoderKL
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/current.yaml")
args, unknown = parser.parse_known_args()
torch.cuda.empty_cache()
device = 'cuda'


# %%
# ==================== Accept Config from Args ====================
parser.add_argument("--codename", type=str, default='auto')
parser.add_argument("--extractor", type=str, default='pretrained/dec_48b_whit.torchscript.pt')
parser.add_argument("--img_size", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_imgs", type=int, default=None)
parser.add_argument("--msg_each", type=int, default=32)
parser.add_argument("--msg_in_order", type=str2bool, default=True)
parser.add_argument("--norm_alpha", type=float, default=1)
parser.add_argument("--norm_epsilon", type=int, default=40)
parser.add_argument("--adapt_alpha_epsilon", type=str2bool, default=True)
parser.add_argument("--min_iter", type=int, default=1)
parser.add_argument("--max_iter", type=int, default=150)
parser.add_argument("--acc_thres", type=float, default=1)
parser.add_argument("--split_pos", type=str, help="Split position - either 'z', 'a0' or an integer", default='z')
parser.add_argument("--bit_length", type=int, default=5)
parser.add_argument("--store", type=str2bool, default=False, help="Enable storing")
parser.add_argument("--saving", type=str2bool, default=False, help="Enable saving")
parser.add_argument("--store_limit", type=str, default='inf', 
                    help="inf for no limit, number for x gb")
# parser.add_argument("--ark_dir", type=str, default='output_ark')
parser.add_argument("--ark_dir", type=str, default='output_turbo')
args, unknown = parser.parse_known_args()

CODENAME = args.codename
EXTRACTOR = args.extractor
IMG_SIZE = args.img_size
BATCH_SIZE = args.batch_size
NUM_IMGS = args.num_imgs
MSG_EACH = args.msg_each
MSG_IN_ORDER = args.msg_in_order
NORM_ALPHA = args.norm_alpha
NORM_EPSILON = args.norm_epsilon
ADAPT_ALPHA_EPSILON = args.adapt_alpha_epsilon
MIN_ITER = args.min_iter
MAX_ITER = args.max_iter
ACC_THRES = args.acc_thres
SPLIT_POS = args.split_pos
BIT_LENGTH = args.bit_length
STORE = args.store
SAVING = args.saving
STORE_LIMIT = args.store_limit
ARK_DIR = args.ark_dir

# TODO the more default
MODEL_ID = '/home/asc/repo/cache/sdxl-turbo'
TORCH_DTYPE = torch.float32
TORCH_DTYPE_STR = 'float32'
LOCAL_FILES_ONLY = True
SEED = 42
LOSS_I = 'watson-vgg'

# >> handler
if CODENAME == 'auto':
    if EXTRACTOR=='pretrained/dec_48b_whit.torchscript.pt':
        CODENAME = 'H'
    elif EXTRACTOR=='pretrained/sstamp.torchscript.pt':
        CODENAME = 'S'
    else:
        raise NotImplementedError
    CODENAME += f'{IMG_SIZE}_A{SPLIT_POS}_B{BIT_LENGTH}'
SPLIT_POS = int(SPLIT_POS) if SPLIT_POS.isdigit() else SPLIT_POS
timestamp = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
output_dir = f'{ARK_DIR}/{timestamp}_{CODENAME}'
sub_bit_range = range(BIT_LENGTH)
utils.seed_everything(SEED)
STORE_LIMIT_GB = None if STORE_LIMIT == 'inf' else int(STORE_LIMIT)

# %%
# ==================== Get ori vae ====================

# # for sanity check
# split_pos = random.randint(0, 28);
# norm_alpha = 0
# max_iter = 1

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/store', exist_ok=True)
os.makedirs(output_dir + '/saving', exist_ok=True)
os.makedirs(output_dir + '/comparison', exist_ok=True)
# print all config above
print(f'> {CODENAME}:')
print(f'    extractor: {EXTRACTOR}')
print(f'    split_pos: {SPLIT_POS}')
print(f'    img_size: {IMG_SIZE}')
print(f'    batch_size: {BATCH_SIZE}')
print(f'    norm_alpha: {NORM_ALPHA}')
print(f'    norm_epsilon: {NORM_EPSILON}')
print(f'    min_iter: {MIN_ITER}')
print(f'    max_iter: {MAX_ITER}')
print(f'    acc_thres: {ACC_THRES}')
print(f'    num_imgs: {NUM_IMGS}')
print()

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    variant="fp16" if TORCH_DTYPE_STR=='float16' else None,
    local_files_only=LOCAL_FILES_ONLY
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
vae_ori: AutoencoderKL = pipe.vae
vae_ori = vae_ori.to(device)
del pipe

vqgan_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    utils_img.normalize_vqgan,
])

# %%

# load latents to cpu
# latents = torch.load(utils.latents_filename(conf.model_id, IMG_SIZE), map_location='cpu')
# latents_val = torch.load(utils.latents_filename(conf.model_id, IMG_SIZE)+'_val', map_location='cpu')
train_loader = utils.latents_dataloader_repeat(
    utils.latents_filename(MODEL_ID, IMG_SIZE),
    BATCH_SIZE,
    num_imgs=NUM_IMGS,
    num_workers=8, collate_fn=None)
# val_loader = utils.latents_dataloader_repeat(
#     utils.latents_filename(conf.model_id, IMG_SIZE)+'_val',
#     conf.val_batch_size,
#     num_imgs=None,
#     num_workers=4, collate_fn=None)
vae_ori.encoder = None
# > get hidden extractor
# if '.torchscript.' in EXTRACTOR:
msg_decoder = (torch.jit.load(EXTRACTOR)
                .to(device))
is_sstamp = 'sstamp' in EXTRACTOR
before_msg_decoder = transforms.Compose(
    [utils_img.unnormalize_vqgan] + 
    ([utils_img.normalize_img] if is_sstamp else [])
)
if is_sstamp:  # checking
    assert IMG_SIZE == 400
    assert BATCH_SIZE%4 == 0
if is_sstamp:
    msg_decoder_model = msg_decoder
    def split4(input: torch.Tensor):
        assert input.shape[0]%4 == 0
        dummy_secret = torch.zeros((4, 100), device=device)
        input_ = input.permute(0, 2, 3, 1)
        batch4s = input_.view(input.shape[0]//4, 4, input_.shape[1], input_.shape[2], input_.shape[3])
        message4s = []
        for batch4 in batch4s:
            sstamp, res, message4 = msg_decoder_model(dummy_secret, batch4)
            # message4 = message4[:, :BIT_LENGTH]
            message4 = message4[:, sub_bit_range]
            message4s.append(message4)
        return torch.cat(message4s, dim=0)
    msg_decoder = split4
else:
    # select sub bit range
    msg_decoder_model = msg_decoder
    def sub_extractor(input: torch.Tensor):
        return msg_decoder_model(input)[:, sub_bit_range]
    msg_decoder = sub_extractor
# > get lossw
def loss_w(
    decoded, keys, 
    temp=0.5 if is_sstamp else 10.0
):
    return F.binary_cross_entropy_with_logits(
        decoded * temp, keys, reduction='mean')
# > get lossi
from loss.loss_provider import LossProvider
import lpips
if LOSS_I == 'mse':
    def loss_i(imgs_w, imgs):
        return torch.mean((imgs_w - imgs) ** 2)
elif LOSS_I == 'watson-dft':
    provider = LossProvider()
    loss_percep = provider.get_loss_function(
        'Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
    loss_percep = loss_percep.to(device)

    def loss_i(imgs_w, imgs):
        return loss_percep(
            (1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
elif LOSS_I == 'watson-vgg':
    provider = LossProvider()
    loss_percep = provider.get_loss_function(
        'Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
    loss_percep = loss_percep.to(device)

    def loss_i(imgs_w, imgs):
        return loss_percep(
            (1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
elif LOSS_I == 'ssim':
    provider = LossProvider()
    loss_percep = provider.get_loss_function(
        'SSIM', colorspace='RGB', pretrained=True, reduction='sum')
    loss_percep = loss_percep.to(device)

    def loss_i(imgs_w, imgs):
        return loss_percep(
            (1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
elif LOSS_I == 'lpips':
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)
    def loss_i(imgs_w, imgs):  # want [-1,1]
        return torch.mean(lpips_alex(imgs_w, imgs))
else:
    raise NotImplementedError

torch.cuda.empty_cache()
gc.collect()

# %% 
# ============================== Monkey Patch ==============================
# monkey patch the forward func of vae.decoder
def pass_full(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    # middle
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    sample = sample.to(upscale_dtype)
    # up
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample
def remain_full(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    return sample

# after conv_act
def pass_conv_act(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    # middle
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    sample = sample.to(upscale_dtype)
    # up
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    return sample
def remain_conv_act(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    sample = self.conv_out(sample, maps=maps)
    return sample

# after up3
def pass_up3(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    # middle
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    sample = sample.to(upscale_dtype)
    # up
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds, maps=maps)
    return sample
def remain_up3(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample

# all_acts[-2]
def pass_aminus2(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    # middle
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    sample = sample.to(upscale_dtype)
    # up
    sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    # for resnet in self.resnets:
    #     sample = resnet(sample, temb=None, scale=1.0, maps=maps)
    sample = self.up_blocks[3].resnets[0](sample, temb=None, scale=1.0, maps=maps)
    sample = self.up_blocks[3].resnets[1](sample, temb=None, scale=1.0, maps=maps)
    # sample = self.up_blocks[3].resnets[2](sample, temb=None, scale=1.0, maps=maps)
    aux = sample.detach()
    sample = self.up_blocks[3].resnets[2](sample, None, maps=maps, aux=aux, split=('out', 0))
    return sample, aux
def remain_aminus2(
    self, sample: torch.FloatTensor, aux: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # skip norm1 & nonlinearity1 for the residual (resnet2)
    sample = self.up_blocks[3].resnets[2](sample, None, maps=maps, aux=aux, split=('in', 0))
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample

# after up2
def pass_up2(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    # middle
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    sample = sample.to(upscale_dtype)
    # up
    sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    return sample
def remain_up2(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample
# TODO
# after conv_in
def pass_conv_in(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    return sample
def remain_conv_in(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = sample.to(upscale_dtype)
    # up
    sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample


# after conv_ina
def pass_conv_ina(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    sample = self.conv_in(sample, maps=maps)
    aux = sample.detach()  # aux: the original input (of resnet)
    sample = self.mid_block.resnets[0](sample, None, maps=maps, split=('out', 0))
    return sample, aux
def remain_conv_ina(
    self, sample: torch.FloatTensor, aux: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    sample = self.mid_block.resnets[0](sample, None, maps=maps, aux=aux, split=('in', 0))
    sample = self.mid_block.attentions[0](sample, temb=None)
    sample = self.mid_block.resnets[1](sample, None, maps=maps)
    # done mid
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = sample.to(upscale_dtype)
    # up
    sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample

def final_split(
    self, sample: torch.FloatTensor, aux: torch.FloatTensor = None, 
    latent_embeds: Optional[torch.FloatTensor] = None, maps = None, 
    split = ('A', 1)  # ('A'/'B' part, act_index)
) -> torch.FloatTensor:
    if not (split[0] == 'B' and split[1] >= 0):
        sample = self.conv_in(sample, maps=maps)
    if split == ('A', 0):
        aux = sample.detach()  # aux: the original input (of resnet)
    if not (split[0] == 'B' and split[1] >= 0):
        sample = self.mid_block.resnets[0](sample, None, maps=maps, split=('out', 0))
    if split == ('A', 0):
        return sample, aux
    if not (split[0] == 'B' and split[1] >= 1):
        sample = self.mid_block.resnets[0](sample, None, maps=maps, aux=aux, split=('in', 0))
        sample = self.mid_block.attentions[0](sample, temb=None)
    sample = self.mid_block.resnets[1](sample, None, maps=maps)
    # done mid
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = sample.to(upscale_dtype)
    # up
    sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample


def splitA(
    self, sample: torch.FloatTensor,
    latent_embeds: Optional[torch.FloatTensor] = None, maps = None, 
    act_index = 0
) -> torch.FloatTensor:
    if act_index == 'z':
        return sample
    sample = self.conv_in(sample, maps=maps)
    if act_index == 'conv_in':
        return sample
    # mid
    if act_index in (0, 1):
        aux = sample.detach()  # aux: the original input (of resnet)
        sample = self.mid_block.resnets[0](sample, None, maps=maps, split=('out', act_index))
        return sample
    sample = self.mid_block.resnets[0](sample, None, maps=maps)
    sample = self.mid_block.attentions[0](sample, temb=None)
    if act_index in (2, 3):
        aux = sample.detach()
        sample = self.mid_block.resnets[1](sample, None, maps=maps, split=('out', act_index-2))
        return sample
    sample = self.mid_block.resnets[1](sample, None, maps=maps)
    # done mid
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = sample.to(upscale_dtype)

    # up
    # sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    assert len(self.up_blocks[0].resnets) == 3
    for i, resnet in enumerate(self.up_blocks[0].resnets):
        if act_index in (4+i*2, 5+i*2):
            aux = sample.detach()
            sample = resnet(sample, None, maps=maps, split=('out', act_index%2))
            return sample
        sample = resnet(sample, None, maps=maps)
    sample = self.up_blocks[0].upsamplers[0](sample, maps=maps)
    # sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    assert len(self.up_blocks[1].resnets) == 3
    for i, resnet in enumerate(self.up_blocks[1].resnets):
        if act_index in (10+i*2, 11+i*2):
            aux = sample.detach()
            sample = resnet(sample, None, maps=maps, split=('out', act_index%2))
            return sample
        sample = resnet(sample, None, maps=maps)
    sample = self.up_blocks[1].upsamplers[0](sample, maps=maps)
    # sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    assert len(self.up_blocks[2].resnets) == 3
    for i, resnet in enumerate(self.up_blocks[2].resnets):
        if act_index in (16+i*2, 17+i*2):
            aux = sample.detach()
            sample = resnet(sample, None, maps=maps, split=('out', act_index%2))
            return sample
        sample = resnet(sample, None, maps=maps)
    sample = self.up_blocks[2].upsamplers[0](sample, maps=maps)
    # sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
    assert len(self.up_blocks[3].resnets) == 3
    for i, resnet in enumerate(self.up_blocks[3].resnets):
        if act_index in (22+i*2, 23+i*2):
            aux = sample.detach()
            sample = resnet(sample, None, maps=maps, split=('out', act_index%2))
            return sample
        sample = resnet(sample, None, maps=maps)
    # no upsampler in up_blocks[3]

    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    if act_index == 28:
        return sample
    sample = self.conv_out(sample, maps=maps)
    return sample  # 'image'

def splitB(
    self, sample: torch.FloatTensor, aux: torch.FloatTensor = None,
    latent_embeds: Optional[torch.FloatTensor] = None, maps = None, 
    act_index = 0
) -> torch.FloatTensor:
    if type(act_index) == int and act_index < 28:
        assert aux is not None
    # for strings:
    if act_index in ('z', 'conv_in'):
        if act_index == 'z':
            sample = self.conv_in(sample, maps=maps)
        sample = self.mid_block(sample, latent_embeds, maps=maps)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        sample = sample.to(upscale_dtype)
        # up
        sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
        sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
        sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
        sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, maps=maps)
        return sample
    # TODO support more for string

    # mid
    if act_index < 2:
        sample = self.mid_block.resnets[0](sample, None, maps=maps, aux=aux, split=('in', act_index))
        sample = self.mid_block.attentions[0](sample, temb=None)
    if act_index < 4:
        if act_index < 2:
            sample = self.mid_block.resnets[1](sample, None, maps=maps)
        else:
            sample = self.mid_block.resnets[1](sample, None, maps=maps, aux=aux, split=('in', act_index%2))
            # done mid
            upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
            sample = sample.to(upscale_dtype)

    # up
    # sample = self.up_blocks[0](sample, latent_embeds, maps=maps)
    for i, resnet in enumerate(self.up_blocks[0].resnets):
        if act_index < 6+i*2:
            if act_index < 4+i*2:
                sample = resnet(sample, None, maps=maps)
            else:
                sample = resnet(sample, None, maps=maps, aux=aux, split=('in', act_index%2))
    if act_index < 10:
        sample = self.up_blocks[0].upsamplers[0](sample, maps=maps)
    # sample = self.up_blocks[1](sample, latent_embeds, maps=maps)
    for i, resnet in enumerate(self.up_blocks[1].resnets):
        if act_index < 12+i*2:
            if act_index < 10:
                sample = resnet(sample, None, maps=maps)
            else:
                sample = resnet(sample, None, maps=maps, aux=aux, split=('in', act_index%2))
    if act_index < 16:
        sample = self.up_blocks[1].upsamplers[0](sample, maps=maps)
    # sample = self.up_blocks[2](sample, latent_embeds, maps=maps)
    for i, resnet in enumerate(self.up_blocks[2].resnets):
        if act_index < 18+i*2:
            if act_index < 16:
                sample = resnet(sample, None, maps=maps)
            else:
                sample = resnet(sample, None, maps=maps, aux=aux, split=('in', act_index%2))
    if act_index < 22:
        sample = self.up_blocks[2].upsamplers[0](sample, maps=maps)
    # sample = self.up_blocks[3](sample, latent_embeds, maps=maps)
    for i, resnet in enumerate(self.up_blocks[3].resnets):
        if act_index < 24+i*2:
            if act_index < 22:
                sample = resnet(sample, None, maps=maps)
            else:
                sample = resnet(sample, None, maps=maps, aux=aux, split=('in', act_index%2))
    # no upsampler in up_blocks[3]

    # post-process
    if act_index < 28:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample  # 'image'


# after z
def pass_z(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    return sample
def remain_z(
    self, sample: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None, maps = None
) -> torch.FloatTensor:
    # conv_in
    sample = self.conv_in(sample, maps=maps)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # mid
    sample = self.mid_block(sample, latent_embeds, maps=maps)
    sample = sample.to(upscale_dtype)
    # up
    for up_block in self.up_blocks:
        sample = up_block(sample, latent_embeds, maps=maps)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample, maps=maps)
    return sample

# ==================== Monkey Split ====================
# match split_pos:
#     case 'full':
#         vae_ori.decoder.forward = pass_full.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_full.__get__(vae_ori.decoder)
#     case 'conv_act' | 'am1':  # -1
#         vae_ori.decoder.forward = pass_conv_act.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_conv_act.__get__(vae_ori.decoder)
#     case 'up3':
#         vae_ori.decoder.forward = pass_up3.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_up3.__get__(vae_ori.decoder)
#     case 'aminus2' | 'am2':  # -2
#         vae_ori.decoder.forward = pass_aminus2.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_aminus2.__get__(vae_ori.decoder)
#     case 'up2':
#         vae_ori.decoder.forward = pass_up2.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_up2.__get__(vae_ori.decoder)
#     # TODO
#     case 'conv_in':
#         vae_ori.decoder.forward = pass_conv_in.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_conv_in.__get__(vae_ori.decoder)
#     case 'conv_ina' | 'a0':  # 0
#         vae_ori.decoder.forward = pass_conv_ina.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_conv_ina.__get__(vae_ori.decoder)
#     case 'z':
#         vae_ori.decoder.forward = pass_z.__get__(vae_ori.decoder)
#         vae_ori.decoder.remain = remain_z.__get__(vae_ori.decoder)
vae_ori.decoder.forward = (lambda self, sample, latent_embeds=None, maps=None: 
    splitA(self, sample, latent_embeds, maps, act_index=SPLIT_POS)).__get__(vae_ori.decoder)
vae_ori.decoder.remain = (lambda self, sample, aux=None, latent_embeds=None, maps=None: 
    splitB(self, sample, aux, latent_embeds, maps, act_index=SPLIT_POS)).__get__(vae_ori.decoder)



# %%
# Initialize empty tensors with correct dimensions
store_filenames = []
using_aux = False
first_time = True
# run start!
for step, z in tqdm(enumerate(train_loader),
                    desc='Training', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(train_loader)):
    print(f'\n\n> step: {step}')
    z = z[0]  # due to using_latents
    z = z.to(device, non_blocking=True)
    z = z.type(TORCH_DTYPE)

    act = vae_ori.decoder(z)
    if type(act) == tuple:
        act, aux = act
        using_aux = True
    act = act.detach()
    y = vae_ori.decoder.remain(act, aux) if using_aux else vae_ori.decoder.remain(act)

    savings = {  # TODO for one now
        'act': act,
        'msg': [],
        'act_adv': [],
        'decoded': [],
        'bit_accs': [],
    }  
    if first_time:
        first_time = False
        print(f'\n> act.shape: {act.shape}\n')

    # random msg
    for msg_idx in range(MSG_EACH):
        if MSG_IN_ORDER:
            # msg = msg_idx to binary, with total length = BIT_LENGTH
            bin_list = list(map(int, bin(msg_idx%2**BIT_LENGTH)[2:]))
            bin_list = [0]*(BIT_LENGTH - len(bin_list)) + bin_list
            msg = torch.tensor(bin_list, dtype=TORCH_DTYPE).to(device, non_blocking=True)
            # repeat to batch_size
            msg = msg.repeat(BATCH_SIZE, 1)
        else:
            msg = torch.randint(0, 2, size=(BATCH_SIZE, BIT_LENGTH),
                            dtype=TORCH_DTYPE,
                            ).to(device, non_blocking=True)
        msg_str = ["".join([str(int(ii)) for ii in msg.tolist()[jj]]) for jj in range(BATCH_SIZE)]
        print(f'- > step {step}-{msg_idx} with batch_size {BATCH_SIZE}, msg_str: {msg_str}')

        # > unNorm for act (which is not in [-1, 1])
        if ADAPT_ALPHA_EPSILON:
            alpha = NORM_ALPHA * act.std()
            epsilon = NORM_EPSILON * act.std()
        else:
            alpha = NORM_ALPHA
            epsilon = NORM_EPSILON
        print(f'norm: alpha: {NORM_ALPHA:.4g}, epsilon: {NORM_EPSILON:.4g}, act.std(): {act.std().item():.4g}')
        print(f'now: alpha: {alpha:.4g}, epsilon: {epsilon:.4g}')
        def l2_step(x, x_adv, grad):
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)
            x_adv = x_adv.detach() - alpha * grad
            delta = x_adv - x
            delta_norm = torch.norm(delta.reshape(delta.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            factor = torch.min(epsilon / (delta_norm + 1e-8), torch.ones_like(delta_norm))
            print(f', factor: {[f"{x:.2f}" for x in factor.flatten().tolist()]}, delta_norm: {[f"{x:.3g}" for x in delta_norm.flatten().tolist()]}')
            delta = delta * factor
            return x+delta
            #! return torch.clamp(x + delta, -1, 1)

        # optimize on act (adversarial) for loss, s.t. l2 on just y(adv later)
        act_adv = act.clone().detach()

        # random init
        # act_adv = act_adv + torch.zeros_like(act_adv).uniform_(-epsilon, epsilon)
        #! act_adv = torch.clamp(act_adv, -1, 1)

        # PGD
        failed = True
        for i in range(MAX_ITER):
            act_adv.requires_grad = True
            y_adv = vae_ori.decoder.remain(act_adv, aux) if using_aux else vae_ori.decoder.remain(act_adv)
            y_adv = torch.clamp(y_adv, -1, 1)
            decoded = msg_decoder(before_msg_decoder(y_adv))  # b c h w -> b k
            lossw = loss_w(decoded, msg)
            if True:
                diff1 = (~torch.logical_xor(decoded > 0, msg > 0))  # b k -> b k
                bit_accs = torch.sum(diff1, dim=-1) / diff1.shape[-1]  # b k -> b
                word_accs = (bit_accs == 1)  # b
            print(f'- {i}: lossw: {lossw.item():.3f}, bit_acc_avg: {torch.mean(bit_accs).item():.3f}, word_acc_avg: {torch.mean(word_accs.type(torch.float)).item():.3f}', end='')
            # after 10 if acc > 0.9, break
            # TODO!!! 采用lossw几个step(10?)不下降作为标准而非acc(除非0.99)
            # if i > min_iter and torch.mean(bit_accs.type(torch.float)).item() > 0.95:
            if i > MIN_ITER and torch.mean(bit_accs.type(torch.float)).item() >= ACC_THRES:
                failed = False
                break
            lossw.backward(retain_graph=True)  # TODO why retain_graph is needed for aux
            grad = act_adv.grad.data.clone().detach()
            act_adv.grad.data.zero_()
            # l2 step
            act_adv = l2_step(act, act_adv, grad)
        if failed:
            print(f'- {i}: failed')
        else:
            print(f'+ {i}: success')

        # ----report----
        act_adv = act_adv.detach()
        delta = act_adv - act; delta = delta.detach().cpu()

        # save this batch
        if STORE:
            start_time = time.time()
            store = {
                'step': step,
                'msg_idx': msg_idx,
                'msg': msg.detach().cpu(),
                # --- key ---
                # 'act': act.detach().cpu(),
                # 'delta': delta.detach().cpu(),
                'act_adv': act_adv.detach().cpu(),
                # 'decoded': decoded.detach().cpu(),
                # --- key ---
                'bit_accs': bit_accs.detach().cpu(),
            }
            torch.save(store, os.path.join(output_dir, f'store/store_{step}_{msg_idx}.pth'))
            store_filenames.append(f'store/store_{step}_{msg_idx}.pth')
            save_time = time.time() - start_time
            print(f'> saved to {os.path.join(output_dir, f"store/store_{step}_{msg_idx}.pth")} in {save_time:.2f}s')

        torch.cuda.empty_cache(); gc.collect()

        img01 = torch.clamp(utils_img.unnormalize_vqgan(y), 0, 1)
        img01_adv = torch.clamp(utils_img.unnormalize_vqgan(y_adv), 0, 1)

        # showing
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        ax1.imshow(img01[0].detach().cpu().numpy().transpose(1, 2, 0))
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(img01_adv[0].detach().cpu().numpy().transpose(1, 2, 0))
        ax2.set_title('Adversarial')
        ax2.axis('off')
        # show diff x20
        diff10 = 0.5 + (img01_adv - img01)*10
        diff10 = torch.clamp(diff10, 0, 1)
        ax3.imshow(diff10[0].detach().cpu().numpy().transpose(1, 2, 0))
        ax3.set_title('Difference (10x)')
        ax3.axis('off')
        # show diff x1
        diff = 0.5 + (img01_adv - img01)
        diff = torch.clamp(diff, 0, 1)
        ax4.imshow(diff[0].detach().cpu().numpy().transpose(1, 2, 0))
        ax4.set_title('Difference (1x)')
        ax4.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison/{step}_{msg_idx}.png'), 
                    bbox_inches='tight', pad_inches=0.1)
        plt.show()

        # to cuda1 before saving
        if SAVING:
            savings['msg'].append(msg.detach().to('cpu'))
            savings['act_adv'].append(act_adv.detach().to('cpu'))
            savings['decoded'].append(decoded.detach().to('cpu'))
            savings['bit_accs'].append(bit_accs.detach().to('cpu'))
        torch.cuda.empty_cache(); gc.collect()

        # if msg_idx+1 == 8:
        #     break

    # if step+1 == 1:
    #     break

    if SAVING:
        torch.save(savings, os.path.join(output_dir, f'saving/saving_{step}.pth'))


# %%
debug = False
if debug:
    savings = torch.load('savings.pth', weights_only=False)
    # TODO and to cuda:1

    # %% for testing savings
    device2 = 'cuda:1'

    torch.cuda.empty_cache(); gc.collect()
    vae_ori.requires_grad_(False)
    vae_ori = vae_ori.to(device2)
    if using_aux:
        aux = aux.to(device2)

    # %%
    # try savings on bin-mixed msg
    same_bits_total = 0
    same_unchange_bits_total = 0
    not_same_bits_total = 0
    not_same_change_bits_total = 0
    for i, (ad1, msg1) in enumerate(zip(
        savings['act_adv'], savings['msg']
    )):
        for j, (ad2, msg2) in enumerate(zip(
            savings['act_adv'], savings['msg']
        )):
            # skip if tested before
            if i >= j:
                continue
            adMix = ad1 + ad2 / 2
            yMix = vae_ori.decoder.remain(adMix, aux) if using_aux else vae_ori.decoder.remain(adMix)

            yMix = yMix.to(device)
            decodedMix = msg_decoder(before_msg_decoder(yMix))  # b c h w -> b k
            decodedMix = decodedMix.to(device2)

            msgMix = (decodedMix > 0).type(torch.float)
            # print(f'{i} {j}: {torch.sum(msg1 == msgMix).item()}/{msg1.numel()}')
            # if same msg bits stay the same after mixed
            same_before_bits = msg1==msg2
            same_unchange_bits = msgMix[same_before_bits] == msg1[same_before_bits]
            not_same_change_bits = msgMix[~same_before_bits] != msg1[~same_before_bits]
            print(f'{i} {j}: {torch.sum(same_unchange_bits).item()} / {torch.sum(same_before_bits).item()}, {torch.sum(not_same_change_bits).item()} / {torch.sum(~same_before_bits).item()}')

            # accumulate
            same_bits_total += torch.sum(same_before_bits).item()
            same_unchange_bits_total += torch.sum(same_unchange_bits).item()
            not_same_change_bits_total += torch.sum(not_same_change_bits).item()
            not_same_bits_total += torch.sum(~same_before_bits).item()
    print(f'same_bits_total: {same_bits_total}, same_unchange_bits_total: {same_unchange_bits_total}, ratio: {same_unchange_bits_total/same_bits_total}')
    print(f'not_same_change_bits_total: {not_same_change_bits_total}, not_same_bits_total: {not_same_bits_total}, ratio: {not_same_change_bits_total/not_same_bits_total}')

    # %%
    # 3d-PCA on act_adv
    from sklearn.decomposition import PCA
    import plotly.express as px
    import pandas as pd
    savings['act'] = savings['act'].to(device2)

    # %%
    instance_to_use = 0
    elems = [savings['act_adv'][i][instance_to_use] for i in range(len(savings['act_adv']))]
    labels = [f'{bin(i)[2:]}' for i in range(len(elems))]
    # fill zeros to make the strings same length
    max_len = max([len(label) for label in labels])
    labels = [label.zfill(max_len) for label in labels]
    labels = ['original'] + labels
    points = torch.stack([savings['act'][instance_to_use]]+elems, dim=0)
    points = points.flatten(start_dim=1)
    points = points.cpu().numpy()

    # %%
    # 2D PCA with plotly
    pca2d = PCA(n_components=2)
    points_pca = pca2d.fit_transform(points)

    # Get the PC vectors
    pc_vectors = pca2d.components_
    print("PC1 vector:", pc_vectors[0])
    print("PC2 vector:", pc_vectors[1])

    # %% 
    # show
    df1 = pd.DataFrame(points_pca, columns=['PC1', 'PC2'])
    df1['label'] = labels

    # draw a arrow between those who only diff in one bit
    diff_bit_index = 4
    diff_bit_value = 1<<diff_bit_index
    pair_diff_in_second_bit = [
        (i, j) for i in range(len(df1)) for j in range(i+1, len(df1)) 
        if (i^diff_bit_value) == j
    ]

    fig1 = px.scatter(df1, x='PC1', y='PC2', text='label',
                        title=f'PCA of Adversarial Activations, diff in bit[{diff_bit_index}]',
                        width=800, height=800)
    fig1.update_traces(textposition='top center')
    for pair in pair_diff_in_second_bit:
        # Add arrow annotation
        fig1.add_annotation(
            x=df1['PC1'][pair[1]+1],  # End point
            y=df1['PC2'][pair[1]+1], 
            ax=df1['PC1'][pair[0]+1],  # Start point
            ay=df1['PC2'][pair[0]+1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,  # Arrow style (1-8)
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='red'
        )

    fig1.show()

    # %%
    #! use the same PC vectors for another elem
    with_ori = 1  # 0: no original, 1: with original

    instance2_to_use = 1
    elem2s = [savings['act_adv'][i][instance2_to_use] for i in range(len(savings['act_adv']))]
    points2 = torch.stack([savings['act'][instance_to_use]]+elem2s, dim=0) if with_ori else torch.stack(elem2s, dim=0)
    points2 = points2.flatten(start_dim=1)
    points2 = points2.cpu().numpy()

    points_pca2 = points2 @ pc_vectors.T

    # show
    df2 = pd.DataFrame(points_pca2, columns=['PC1', 'PC2'])
    if with_ori:
        df2['label'] = labels
    else:
        df2['label'] = labels[1:]

    fig2 = px.scatter(df2, x='PC1', y='PC2', text='label',
                        title=f'PCA of Adversarial Activations, diff in bit[{diff_bit_index}], apply on another instance',
                        width=800, height=800)
    fig2.update_traces(textposition='top center')
    for pair in pair_diff_in_second_bit:
        # Add arrow annotation
        fig2.add_annotation(
            x=df2['PC1'][pair[1]+with_ori],  # End point
            y=df2['PC2'][pair[1]+with_ori], 
            ax=df2['PC1'][pair[0]+with_ori],  # Start point
            ay=df2['PC2'][pair[0]+with_ori],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,  # Arrow style (1-8)
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor='red'
        )
    fig2.show()


