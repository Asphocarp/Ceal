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
from diffusers.models.vae import Decoder
from diffusers.models.autoencoder_kl import AutoencoderKL
from pathfinder import Pathfinder, Policy
import argparse
import wandb
from torch import nn
import numpy as np
from tqdm import tqdm
from mapper import MappingNetwork
from PIL import Image
import gc
import misc


# %%==================== Config ====================
parser = argparse.ArgumentParser()
parser.add_argument("--codename", type=str, default=None, help="None for auto-generated")
parser.add_argument("--acc", type=utils.str2bool, default=False, help="use accelerator")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_id", type=str, default='stabilityai/sdxl-turbo')
parser.add_argument("--local_files_only", type=utils.str2bool, default=True)
parser.add_argument("--use_safetensors", type=utils.str2bool, default=True)
parser.add_argument("--torch_dtype_str", type=str, default='float32')
parser.add_argument("--bit_length", type=int, default=48)
parser.add_argument("--hidden_dims", type=str, default='[1024]', 
                    help='hidden dims for mapping network, like [1024,1024]')
parser.add_argument("--vae_dec_ckpt", type=str, default='',
                    help='path to vae.decoder checkpoint')
parser.add_argument("--ex_type", type=str, default='hidden',
                    help='random, hidden, sstamp, jit ... TODO')
parser.add_argument("--ex_ckpt", type=str, default='pretrained/dec_48b_whit.torchscript.pt',
                    help='path to extractor checkpoint')
parser.add_argument("--train_dir", type=str, default='../cache/train2014',
                    help='path to train dataset')
parser.add_argument("--val_dir", type=str, default='../cache/val2014',
                    help='path to val dataset')
parser.add_argument("--output_dir", type=str, default='output_turbo')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--val_batch_size", type=int, default=4)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--val_img_size", type=int, default=256)
parser.add_argument("--val_img_num", type=int, default=1000)
parser.add_argument("--train_ex", type=utils.str2bool, default=False)
parser.add_argument("--distortion", type=utils.str2bool, default=False)
parser.add_argument("--use_cached_latents", type=utils.str2bool, default=False)
parser.add_argument("--lossw", type=str, default='bce')
parser.add_argument("--lossi", type=str, default='watson-vgg')
parser.add_argument("--optimizer", type=str, default='AdamW,lr=1e-4')
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--cosine_lr", type=utils.str2bool, default=False)
parser.add_argument("--consi", type=float, default=2, 
                    help="constraint for lossi, set to -1 to disable")
parser.add_argument("--lambdai", type=float, default=0.2, 
                    help='lambda for lossi')
parser.add_argument("--save_img_freq", type=int, default=50)
parser.add_argument("--save_ckpt_freq", type=int, default=2000)
parser.add_argument("--validate", type=utils.str2bool, default=True)
parser.add_argument("--save_all_ckpts", type=utils.str2bool, default=False)
parser.add_argument("--proj_max_step", type=int, default=10,
                    help='max step for projection')
parser.add_argument("--stop_when_ascent", type=utils.str2bool, default=True,
                    help='stop projectionwhen lossi stop decreasing')
# pathfinder config
parser.add_argument("--granularity", type=str, default='kernel',
                    help='granularity for pathfinder')
parser.add_argument("--layer_selection", type=str, default='layer_range',
                    help='layer selection for pathfinder')
parser.add_argument("--layer_begin", type=str, default='up_blocks.1.resnets.0.conv1',
                    help='layer begin for pathfinder')
parser.add_argument("--layer_end", type=str, default='up_blocks.3.resnets.0.conv1',
                    help='layer end for pathfinder')
parser.add_argument("--use_lora", type=utils.str2bool, default=True)
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--channel_selection", type=str, default='random',
                    help='channel selection for pathfinder')
parser.add_argument("--include_bias", type=utils.str2bool, default=False)
parser.add_argument("--enable_group", type=utils.str2bool, default=True)
parser.add_argument("--continuous_groups", type=utils.str2bool, default=True)
parser.add_argument("--chain", type=utils.str2bool, default=True)
parser.add_argument("--total_group_num", type=int, default=32)
parser.add_argument("--group_num", type=int, default=32)
parser.add_argument("--start_group", type=int, default=11)
parser.add_argument("--conv_out_full_out", type=utils.str2bool, default=False)
parser.add_argument("--conv_in_null_in", type=utils.str2bool, default=True)
parser.add_argument("--absolute_perturb", type=utils.str2bool, default=False)
# if ex_type is hidden
parser.add_argument("--hidden_redundancy", type=int, default=0,
                    help='redundancy for hidden extractor')
parser.add_argument("--hidden_depth", type=int, default=3,
                    help='depth for hidden extractor')
parser.add_argument("--hidden_channels", type=int, default=128,
                    help='channels for hidden extractor')
args, unknown = parser.parse_known_args()

# >> rename
def main(args):
    CODENAME = args.codename
    ACC = args.acc
    SEED = args.seed
    MODEL_ID = args.model_id
    LOCAL_FILES_ONLY = args.local_files_only
    USE_SAFE_TENSORS = args.use_safetensors
    TORCH_DTYPE_STR = args.torch_dtype_str
    BIT_LENGTH = args.bit_length
    HIDDEN_DIMS_STR = args.hidden_dims
    VAE_DEC_CKPT = args.vae_dec_ckpt
    EX_TYPE = args.ex_type
    EX_CKPT = args.ex_ckpt
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    OUTPUT_DIR = args.output_dir
    BATCH_SIZE = args.batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    STEPS = args.steps
    IMG_SIZE = args.img_size
    VAL_IMG_SIZE = args.val_img_size
    VAL_IMG_NUM = args.val_img_num
    TRAIN_EX = args.train_ex
    DISTORTION = args.distortion
    USE_CACHED_LATENTS = args.use_cached_latents
    LOSSW = args.lossw
    LOSSI = args.lossi
    OPTIMIZER = args.optimizer
    LOG_FREQ = args.log_freq
    SAVE_IMG_FREQ = args.save_img_freq
    SAVE_CKPT_FREQ = args.save_ckpt_freq
    WARMUP_STEPS = args.warmup_steps
    COSINE_LR = args.cosine_lr
    CONSI = args.consi
    LAMBDAI = args.lambdai
    VALIDATE = args.validate
    SAVE_ALL_CKPTS = args.save_all_ckpts
    # GreedyPGD (1. loose proj till in eps_i 2. GD with w&i)
    STOP_WHEN_ASCENT = args.stop_when_ascent
    PROJ_MAX_STEP = args.proj_max_step
    # hidden
    HIDDEN_REDUNDANCY = args.hidden_redundancy
    HIDDEN_DEPTH = args.hidden_depth
    HIDDEN_CHANNELS = args.hidden_channels

    # >> handler
    CONF_DICT = vars(args)
    print('> CONF_DICT:\n', CONF_DICT)
    if CODENAME is None:
        CODENAME = misc.time_str('%m%d_%H%M%S')
    TORCH_DTYPE = getattr(torch, TORCH_DTYPE_STR)
    utils.seed_everything(SEED)
    # pure str conf dict
    HIDDEN_DIMS = eval(HIDDEN_DIMS_STR)  # TODO safe_eval / str2list
    # ODIR: output of this run
    ODIR = os.path.join(OUTPUT_DIR, CODENAME)
    CLEAN_MODEL_DIR = os.path.join(OUTPUT_DIR, '_clean_', MODEL_ID.replace('/', '_'))
    # makedirs
    output_subdirs = ['train', 'validate', 'test_caption', 'test_caption_result', 'test_x_svg']
    clean_model_subdirs = ['test_caption', 'caption_latent']
    for subdir in output_subdirs:
        os.makedirs(os.path.join(ODIR, subdir), exist_ok=True)
    for subdir in clean_model_subdirs:
        os.makedirs(os.path.join(CLEAN_MODEL_DIR, subdir), exist_ok=True)


    # %%==================== Main ====================
    # >> init logging
    if ACC:
        from accelerate import Accelerator
        accelerator = Accelerator(log_with='wandb')
        device = accelerator.device
        accelerator.init_trackers(
            project_name="Ceal",
            config=CONF_DICT,
            init_kwargs={"wandb": {"name": CODENAME}})
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        wandb.init(project="Ceal", config=CONF_DICT, name=CODENAME)

    # >> get ori vae
    pipe = utils.get_pipe(
        model_id=MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
        use_safetensors=USE_SAFE_TENSORS,
        torch_dtype=TORCH_DTYPE,
    )
    vae_ori: AutoencoderKL = pipe.vae
    vae_ori: AutoencoderKL = vae_ori.to(device)
    decoder_ori: Decoder = vae_ori.decoder

    # >> copy, get path, init mappers
    decoder_tune: Decoder = deepcopy(decoder_ori)
    decoder_tune = decoder_tune.to(device)
    # with pathfinder
    po = Policy(**CONF_DICT)  # TODO verify dict
    pf = Pathfinder(po)
    pf.explore(decoder_tune)
    pf.print_path()
    total_size = pf.init_model(decoder_tune)
    print(f'> total_size: {total_size}')
    if ACC: accelerator.log({'total_size': total_size}, step=0)
    else: wandb.log({'total_size': total_size}, step=0)
    mn = MappingNetwork(
        bit_length=BIT_LENGTH,
        output_size=total_size,
        hidden_dims=HIDDEN_DIMS,
        device=device,
    )
    # # pipeline parallel (NOT USING NOW)
    # if PIPELINE_PARALLEL:
    #     from parallel import pipeline_parallel_decoder
    #     pipeline_parallel_decoder(
    #         decoder_tune,
    #         gpu_num=conf.parallel_gpu_num,
    #         split_size=conf.parallel_split_size,
    #     )

    # >> load vae.decoder ckpt (actually mn ckpt now)
    if VAE_DEC_CKPT != '':
        checkpoint = torch.load(VAE_DEC_CKPT, map_location='cpu')
        # fix state dict of mn
        # replace all key like 'module.seq.0.weight' to 'seq.0.weight'
        fixed_state_dict = {}
        for key, value in checkpoint['mapping_network'].items():
            # Check if the key starts with 'module.' and replace it
            if key.startswith('module.'):
                new_key = key.replace('module.', '', 1)
            else:
                new_key = key
            fixed_state_dict[new_key] = value
        mn.load_state_dict(checkpoint['mapping_network'])
        mn = mn.to(device)

    # >> get ex
    if EX_TYPE in ['hidden', 'jit', 'sstamp']:
        if '.torchscript.' in EX_CKPT:
            msg_decoder = torch.jit.load(EX_CKPT).to(device)
        else:
            msg_decoder = utils.get_hidden_decoder(
                num_bits=BIT_LENGTH,
                redundancy=HIDDEN_REDUNDANCY,
                num_blocks=HIDDEN_DEPTH,
                channels=HIDDEN_CHANNELS).to(device)
            # use pretrained decoder
            ckpt = utils.get_hidden_decoder_ckpt(EX_CKPT)
            print(msg_decoder.load_state_dict(ckpt, strict=False))
            msg_decoder.eval()
            # whitening
            print(f'>>> Whitening...')
            with torch.no_grad():
                # features from the dataset
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225]),
                ])
                loader = utils.get_dataloader(
                    TRAIN_DIR, transform, batch_size=16, collate_fn=None)
                ys = []
                for i, x in tqdm(enumerate(loader), total=len(loader)):
                    x = x.to(device)
                    y = msg_decoder(x)
                    ys.append(y.to('cpu'))
                ys = torch.cat(ys, dim=0)
                nbit = ys.shape[1]
                # whitening
                mean = ys.mean(dim=0, keepdim=True)  # NxD -> 1xD
                ys_centered = ys - mean  # NxD
                cov = ys_centered.T @ ys_centered
                e, v = torch.linalg.eigh(cov)
                L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
                weight = torch.mm(L, v.T)
                bias = -torch.mm(mean, weight.T).squeeze(0)
                linear = nn.Linear(nbit, nbit, bias=True)
                linear.weight.data = np.sqrt(nbit) * weight
                linear.bias.data = np.sqrt(nbit) * bias
                msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
                torchscript_m = torch.jit.script(msg_decoder)
                EX_CKPT = EX_CKPT.replace(".pth", "_whit.torchscript.pth")
                print(f'>>> Creating torchscript at {EX_CKPT}...')
                torch.jit.save(torchscript_m, EX_CKPT)
    elif EX_TYPE == 'random':
        print(f'>>> Using random extractor...')
        msg_decoder = utils.get_hidden_decoder(
            num_bits=BIT_LENGTH,
            redundancy=HIDDEN_REDUNDANCY,
            num_blocks=3,
            channels=128)
        def shuffle_params(m):
            if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
                param = m.weight
                m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())
                param = m.bias
                m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))
        shuffle_params(msg_decoder)
        msg_decoder = msg_decoder.to(device)
    elif EX_TYPE == 'resnet':
        from torchvision.models import resnet50, ResNet50_Weights
        msg_decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        msg_decoder.fc = torch.nn.Linear(2048,32)
        msg_decoder = msg_decoder.to(device)
        msg_decoder.load_state_dict(torch.load(EX_CKPT, map_location='cpu'))
    else:
        raise NotImplementedError

    # >> prepare to train
    vae_ori.eval()
    for param in [*vae_ori.parameters()]:
        param.requires_grad = False
    decoder_tune.eval().requires_grad_(False)
    for param in [*msg_decoder.parameters()]:
        param.requires_grad = TRAIN_EX
    # for sstamp: make real msg_decoder func for 
    if 'sstamp' in EX_CKPT:
        msg_decoder_model = msg_decoder
        def aux(input: torch.Tensor):
            # input: now in imgnet space (normalized), like -2.2~2.2
            dummy_secret = torch.zeros((input.shape[0], BIT_LENGTH), device=device)
            # make sure input is in [0, 1]
            input_ = utils_img.unnormalize_img(input)
            # from BCHW to BHWC
            input_ = input_.permute(0, 2, 3, 1)
            sstamp, res, message = msg_decoder_model(dummy_secret, input_)
            return message
        msg_decoder = aux
    # mappers
    all_mapper_params = []
    all_mapper_params += list(mn.parameters())
    mn.train().requires_grad_(True)
    # loader and transforms
    vqgan_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    val_vqgan_transform = transforms.Compose([
        transforms.Resize(VAL_IMG_SIZE),
        transforms.CenterCrop(VAL_IMG_SIZE),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    if not USE_CACHED_LATENTS:
        train_loader = utils.get_dataloader_then_repeat(
            TRAIN_DIR, vqgan_transform, BATCH_SIZE,
            num_imgs=BATCH_SIZE * STEPS,
            shuffle=True, num_workers=4, collate_fn=None)
        val_loader = utils.get_dataloader(
            VAL_DIR, val_vqgan_transform, VAL_BATCH_SIZE,
            num_imgs=VAL_BATCH_SIZE * VAL_IMG_NUM,
            shuffle=False, num_workers=4, collate_fn=None)
    else:
        train_loader = utils.latents_dataloader_repeat(
            utils.latents_filename(MODEL_ID, IMG_SIZE),
            BATCH_SIZE,
            num_imgs=BATCH_SIZE * STEPS,
            num_workers=4, collate_fn=None)
        val_loader = utils.latents_dataloader_repeat(
            utils.latents_filename(MODEL_ID, VAL_IMG_SIZE)+'_val',
            VAL_BATCH_SIZE,
            num_imgs=VAL_IMG_NUM,
            num_workers=4, collate_fn=None)
        vae_ori.encoder = None
        gc.collect()
    vqgan_to_imnet = transforms.Compose(
        [utils_img.unnormalize_vqgan, utils_img.normalize_img])
    if DISTORTION:  # for additional robustness when training msg_decoder
        before_msg_decoder = transforms.Compose([
            vqgan_to_imnet,
            transforms.RandomErasing(),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.95, 1.0))])
    else: before_msg_decoder = transforms.Compose([vqgan_to_imnet])
    
    # >> losses
    # lossw
    if LOSSW == 'bce':
        def loss_w(decoded, keys, temp=1.0 if 'sstamp' in EX_CKPT else 10.0):
            return F.binary_cross_entropy_with_logits(
                decoded * temp, keys, reduction='mean')
    else: raise NotImplementedError
    # lossi
    if LOSSI == 'mse':
        def loss_i(imgs_w, imgs): return torch.mean((imgs_w - imgs) ** 2)
    elif LOSSI == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            'Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        def loss_i(imgs_w, imgs):
            return loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif LOSSI == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            'Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        def loss_i(imgs_w, imgs):
            return loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif LOSSI == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            'SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        def loss_i(imgs_w, imgs):
            return loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif LOSSI == 'lpips':
        lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)
        def loss_i(imgs_w, imgs):  # want [-1,1]
            return torch.mean(lpips_alex(imgs_w, imgs))
    else: raise NotImplementedError

    # >> acc for models
    if ACC: decoder_tune, mn, vae_ori, msg_decoder = accelerator.prepare(decoder_tune, mn, vae_ori, msg_decoder)

    # >> optimizer
    to_optim = all_mapper_params
    if TRAIN_EX:
        to_optim += list(msg_decoder.parameters())
    optim_params = utils.parse_params(OPTIMIZER)
    optimizer = utils.build_optimizer(model_params=to_optim, **optim_params)
    if ACC: train_loader, optimizer = accelerator.prepare(train_loader, optimizer)

    # >> train
    header = 'train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    base_lr = optimizer.param_groups[0]["lr"]
    if ACC: base_lr *= accelerator.num_processes
    no_w_step, ascent_step, proj_max_step = 0, 0, 0
    # loop
    for step, imgs_in in enumerate(metric_logger.log_every(train_loader, LOG_FREQ, header)):
        if USE_CACHED_LATENTS: imgs_in = imgs_in[0]  # no label
        imgs_in = imgs_in.to(device, non_blocking=True)
        imgs_in = imgs_in.type(TORCH_DTYPE)
        utils.adjust_learning_rate(
            optimizer, step, STEPS, WARMUP_STEPS, base_lr, cosine_lr=COSINE_LR)
        optimizer.zero_grad()

        # gen msg
        msg = torch.randint(0, 2, size=(BATCH_SIZE, BIT_LENGTH), dtype=TORCH_DTYPE).to(device, non_blocking=True)
        msg_str = ["".join([str(int(ii)) for ii in msg.tolist()[jj]]) for jj in range(BATCH_SIZE)]

        # gen image
        z = vae_ori.pass_post_quant_conv(imgs_in) if not USE_CACHED_LATENTS else imgs_in
        imgs_res = vae_ori.decoder(z)  # TODO load cached res as well
        flat_maps = mn(msg)
        imgs_w = decoder_tune(z, maps=flat_maps)
        imgs_compare = imgs_res
        lossi = loss_i(imgs_w, imgs_compare)

        to_project = CONSI > 0 and lossi > CONSI
        if not to_project:
            decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
            lossw = loss_w(decoded, msg)
            loss = lossw + LAMBDAI * lossi
            loss.backward()
            optimizer.step()
        else:  # proj
            proj_step_counter = 0  # for PROJ_MAX_STEP
            last_lossi = lossi  # for STOP_WHEN_ASCENT
            # break when 1. done descent 2. ascent 3. max iter
            while lossi > CONSI and (not STOP_WHEN_ASCENT or last_lossi>=lossi) and proj_step_counter < PROJ_MAX_STEP:
                last_lossi = lossi
                # > op.apply lossi
                (LAMBDAI*lossi).backward(retain_graph=False)
                optimizer.step()
                optimizer.zero_grad()
                # > get lossi
                flat_maps = mn(msg)
                imgs_w = decoder_tune(z, maps=flat_maps)
                lossi = loss_i(imgs_w, imgs_compare)
                print(f'> going in => lossi: {lossi}')
                proj_step_counter += 1
                no_w_step += 1  # acctually accumulated w step (on same batch)
            # for case 2&3
            if lossi > CONSI:
                # case 2: ascent
                if last_lossi < lossi:
                    ascent_step += 1
                    print(f'> given up for increasing lossi: {last_lossi} -> {lossi}')
                    del last_lossi
                    with torch.no_grad():  # just for logging w
                        decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                        lossw = loss_w(decoded, msg)
                        loss = lossw + LAMBDAI * lossi
                # case 3: max iter
                elif proj_step_counter >= PROJ_MAX_STEP:
                    proj_max_step += 1
                    print(f'> given up for MAX ITER {proj_step_counter}, lossi: {lossi}')
                    with torch.no_grad():  # just for logging w
                        decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                        lossw = loss_w(decoded, msg)
                        loss = lossw + LAMBDAI * lossi
                optimizer.zero_grad()  # for sure
            # for case 1: done proj
            else: 
                print(f'> loose in after {proj_step_counter}, lossi: {lossi}')
                # go one w step after going in (since we got i, getting w is cheap)
                decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                lossw = loss_w(decoded, msg)
                loss = lossw + LAMBDAI * lossi
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  # for sure

        # log stats
        diff1 = (~torch.logical_xor(decoded > 0, msg > 0))  # b k -> b k
        bit_accs = torch.sum(diff1, dim=-1) / diff1.shape[-1]  # b k -> b
        word_accs = (bit_accs == 1)  # b
        log_stats = {
            "loss": loss.item(),
            "loss_w": lossw.item(),
            "loss_i": lossi.item(),
            "psnr": utils_img.psnr(imgs_w, imgs_compare).mean().item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"],
            "no_w_step": no_w_step,
            "step": step,
            "ascent_step": ascent_step,
            "proj_max_step": proj_max_step,
        }

        for log_key, log_val in log_stats.items():
            metric_logger.update(**{log_key: log_val})

        if ACC: accelerator.log(log_stats, step=step * accelerator.num_processes)
        else: wandb.log(log_stats, step=step)

        # save images during training
        if (step + 1) % SAVE_IMG_FREQ == 0:
            if not ACC or accelerator.is_main_process:
                if not USE_CACHED_LATENTS:
                    save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_in), 0, 1),
                        os.path.join(ODIR, 'train', f'{step:05}_{header}_orig.png'), nrow=8)
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_res), 0, 1),
                    os.path.join(ODIR, 'train', f'{step:05}_{header}_res.png'), nrow=8)
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                    os.path.join(ODIR, 'train', f'{step:05}_{header}_w.png'), nrow=8)

        # remove all intermediate objects
        del imgs_in, z, imgs_res, flat_maps, imgs_w, imgs_compare, lossi, decoded, lossw, loss, diff1, bit_accs, word_accs

        # validate and save checkpoint
        if (step + 1) % SAVE_CKPT_FREQ == 0:  # or step == 0
            # validate
            if VALIDATE:
                decoder_tune.eval()
                val_stats = val(
                    device, vae_ori, decoder_tune, msg_decoder, val_loader, vqgan_to_imnet, mn, step,
                    IMG_SIZE, VAL_BATCH_SIZE, BIT_LENGTH, LOG_FREQ, USE_CACHED_LATENTS, TORCH_DTYPE,
                    EX_CKPT, SAVE_IMG_FREQ, ODIR)
                if ACC: accelerator.log(val_stats, step=step * accelerator.num_processes)
                else: wandb.log(val_stats, step=step)
                decoder_tune.train()
            # save the latest checkpoint
            if not ACC or accelerator.is_main_process:
                if ACC:
                    dict1 = {
                        # 'decoder_tune': decoder_tune.state_dict(),
                        # 'msg_decoder': msg_decoder.state_dict(),
                        'mapping_network': accelerator.unwrap_model(mn).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'params': dict(CONF_DICT),
                    }
                    if SAVE_ALL_CKPTS: accelerator.save(dict1, os.path.join(ODIR, f"ckpt_{step+1}.pth"))
                    else: accelerator.save(dict1, os.path.join(ODIR, f"ckpt.pth"))
                else:
                    dict1 = {
                        # 'decoder_tune': decoder_tune.state_dict(),
                        # 'msg_decoder': msg_decoder.state_dict(),
                        'mapping_network': mn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'params': dict(CONF_DICT),
                    }
                    if SAVE_ALL_CKPTS: torch.save(dict1, os.path.join(ODIR, f"ckpt_{step+1}.pth"))
                    else: torch.save(dict1, os.path.join(ODIR, f"ckpt.pth"))

    print("Averaged {} stats:".format('train'), metric_logger)
    # train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if ACC: accelerator.end_training()
    torch.cuda.empty_cache()


@torch.no_grad()
def val(
    device, vae_ori, decoder_tune, msg_decoder, val_loader, vqgan_to_imnet, mn, train_step,
    IMG_SIZE, VAL_BATCH_SIZE, BIT_LENGTH, LOG_FREQ, USE_CACHED_LATENTS, TORCH_DTYPE,
    EX_CKPT, SAVE_IMG_FREQ, ODIR,
):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    # for sstamp:
    sstamp_resize = transforms.Compose([ transforms.Resize(IMG_SIZE) ])
    for ii, imgs_in in enumerate(metric_logger.log_every(val_loader, LOG_FREQ, header)):
        if USE_CACHED_LATENTS: imgs_in = imgs_in[0]  # no label
        imgs_in = imgs_in.to(device, non_blocking=True)
        imgs_in = imgs_in.type(TORCH_DTYPE)
        # gen msg
        msg_val = torch.randint(0, 2, size=(VAL_BATCH_SIZE, BIT_LENGTH), dtype=TORCH_DTYPE).to(device, non_blocking=True)

        # gen image
        z = vae_ori.pass_post_quant_conv(imgs_in) if not USE_CACHED_LATENTS else imgs_in
        imgs_res = vae_ori.decoder(z)
        flat_maps = mn(msg_val)
        imgs_w = decoder_tune(z, maps=flat_maps)

        log_stats = {
            "psnr_val": utils_img.psnr(imgs_w, imgs_res).mean().item(),
        }
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_07': lambda x: utils_img.resize(x, 0.7),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            # for sstamp, ensure resize to 400, 400
            if 'sstamp' in EX_CKPT: imgs_aug = sstamp_resize(imgs_aug)
            decoded = msg_decoder(imgs_aug)  # b c h w -> b k
            diff = (~torch.logical_xor(decoded > 0, msg_val > 0))  # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
            word_accs = (bit_accs == 1)  # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        # mkdir
        os.makedirs(os.path.join(ODIR, 'validate', f'{train_step}'), exist_ok=True)

        if ii % SAVE_IMG_FREQ == 0:
            orig_filename = os.path.join(ODIR, 'validate', f'{train_step}', f'{ii:05}_val_orig.png')
            res_filename = os.path.join(ODIR, 'validate', f'{train_step}', f'{ii:05}_val_res.png') 
            w_filename = os.path.join(ODIR, 'validate', f'{train_step}', f'{ii:05}_val_w.png')
            diff_filename = os.path.join(ODIR, 'validate', f'{train_step}', f'{ii:05}_val_zdiff.png')
            if not USE_CACHED_LATENTS:
                save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_in), 0, 1), orig_filename, nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_res), 0, 1), res_filename, nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1), w_filename, nrow=8)
            # save the diff*10 between res and w
            try:  # TODO optimize
                img_1 = Image.open(res_filename)
                img_2 = Image.open(w_filename)
                diff = np.abs(np.asarray(img_1).astype(int) - np.asarray(img_2).astype(int)) * 10
                diff = Image.fromarray(diff.astype(np.uint8))
                diff.save(diff_filename)
            except: pass

    print("Averaged {} stats:".format('eval'), metric_logger)
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return val_stats


if __name__ == '__main__':
    main(args)