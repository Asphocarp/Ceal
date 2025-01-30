#!/usr/bin/env python
# %%
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan_xl'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan_xl', 'pg_modules'))
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
import numpy as np
from tqdm import tqdm
from mapper import MappingNetwork
from PIL import Image
from misc import all_exist
import click
import pandas as pd
import shutil
from pathlib import Path
from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, DiTPipeline

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


@click.group(context_settings={"ignore_unknown_options": True})
@click.pass_context
def cli(ctx):
    pass

@cli.command(context_settings={"ignore_unknown_options": True})
@click.pass_context
@click.option('--ckpt', default='outputs/random/rand1-0.pth',)
@click.option('--test_dir', default='../cache/val2014',)
@click.option('--anno', default='../cache/annotations/captions_val2014.json',)
@click.option('--num_imgs', default=5000, help='-1 for all')
@click.option('--test_img_size', default=512,)
@click.option('--test_batch_size', default=4,)
@click.option('--overwrite', default=False,)
@click.option('--cli_msg', default='111010110101000001010111010011010100010000100111', help='random for random msg for each image')
# save z, res, w
@click.option('--save_in', default=False,)
@click.option('--save_z_res', default=True,)
@click.option('--save_w', default=True,)
@click.option('--save_in_to', default="../cache/val2014_512")
def gen(ctx, ckpt, test_dir, anno, num_imgs, test_img_size, test_batch_size, overwrite, cli_msg, save_in, save_z_res, save_w, save_in_to):
    """Gen clean&wm images from captions."""
    gen_func(ckpt, test_dir, anno, num_imgs, test_img_size, test_batch_size, overwrite, cli_msg, save_in, save_z_res, save_w, save_in_to)
def gen_func(ckpt, test_dir, anno, num_imgs, test_img_size, test_batch_size, overwrite, cli_msg, save_in, save_z_res, save_w, save_in_to):
    # >> load ckpt (and config)
    checkpoint = torch.load(ckpt, map_location='cpu')
    CONF_DICT = checkpoint['params']

    # >> rename
    # from CONF_DICT
    TORCH_DTYPE_STR = CONF_DICT['torch_dtype_str']
    SEED = CONF_DICT['seed']
    MODEL_ID = CONF_DICT['model_id']
    LOCAL_FILES_ONLY = CONF_DICT['local_files_only']
    USE_SAFETENSORS = CONF_DICT['use_safetensors']
    EX_TYPE = CONF_DICT['ex_type']
    EX_CKPT = CONF_DICT['ex_ckpt']
    BIT_LENGTH = CONF_DICT['bit_length']
    LOSSW = CONF_DICT['lossw']
    LOSSI = CONF_DICT['lossi']
    HIDDEN_DIMS_STR = CONF_DICT['hidden_dims']
    HIDDEN_DIMS = eval(HIDDEN_DIMS_STR)  # TODO safe_eval / str2list

    # >> handler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_DTYPE = getattr(torch, TORCH_DTYPE_STR)
    utils.seed_everything(SEED)
    # model type
    if MODEL_ID.startswith(('stylegan-xl:', 'stylegan3:')):
        model_lib = 'gan'
        # config for gan
        model_url = MODEL_ID.split(':', 1)[1]
        truncation_psi: float = 1.0
        noise_mode: str = 'const'
        class_idx: Optional[int] = None
        centroids_path: Optional[str] = None  # or 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet_centroids.npy'
        Gargs = {'noise_mode': noise_mode}
    else:
        model_lib = 'diffusers'
    ODIR = os.path.dirname(ckpt); print(f'> ODIR: {ODIR}')
    OUTPUT_DIR = os.path.dirname(ODIR)  # the ark
    CLEAN_MODEL_DIR = os.path.join(OUTPUT_DIR, '_clean_', MODEL_ID.replace('/', '_'))
    os.makedirs(CLEAN_MODEL_DIR, exist_ok=True)
    os.makedirs(save_in_to, exist_ok=True)

    # >> get ori model; copy, get path; print total size of proxy
    if model_lib == 'diffusers':
        pipe = utils.get_pipe(
            model_id=MODEL_ID,
            local_files_only=LOCAL_FILES_ONLY,
            use_safetensors=USE_SAFETENSORS,
            torch_dtype=TORCH_DTYPE)
        vae_ori: AutoencoderKL = pipe.vae
        vae_ori: AutoencoderKL = vae_ori.to(device)
        decoder_ori: Decoder = vae_ori.decoder
        decoder_tune: Decoder = deepcopy(decoder_ori)
        decoder_tune = decoder_tune.to(device)
        # with pathfinder
        po = Policy(**CONF_DICT)  # TODO verify dict
        pf = Pathfinder(po)
        pf.explore(decoder_tune)
        pf.print_path()
        total_size = pf.init_model(decoder_tune)
        pipe = pipe.to(device)  # for gen
    elif model_lib == 'gan':
        from stylegan_xl import legacy, dnnlib
        from stylegan_xl.torch_utils import gen_utils
        from stylegan_xl.training.networks_stylegan3_resetting import SynthesisLayer, Generator, SynthesisInput, SynthesisNetwork
        # load model
        with dnnlib.util.open_url(model_url) as f:
            G: Generator = legacy.load_network_pkl(f)['G_ema']
        G: Generator = G.eval().requires_grad_(False).to(device)
        # rebound methods after loading (pathfinder)
        G.__class__.forward = Generator.forward
        G.synthesis.__class__.forward = SynthesisNetwork.forward
        syns = []  # modify Synthesis layers
        rank = CONF_DICT['lora_rank']
        curr = 0  # current pos in flat_maps
        for name, module in G.synthesis.named_children():
            real_class = module._orig_class_name  # due to decorator, the class name is not SynthesisLayer
            if real_class == 'SynthesisLayer':
                module.__class__.forward = SynthesisLayer.forward
                module.granularity = CONF_DICT['granularity']
                module.rank = rank
                match CONF_DICT['granularity']:
                    case 'filter':
                        partition_size = module.out_channels
                    case 'kernel':
                        partition_size = rank * (module.out_channels+module.in_channels)
                    case 'float':
                        partition_size = rank * (module.out_channels + module.in_channels*module.weight.shape[2]*module.weight.shape[3])
                if CONF_DICT['include_bias']:
                    part2_size = module.out_channels
                    module.partition = [
                        curr, curr+partition_size, curr+partition_size+part2_size]
                    curr += partition_size+part2_size
                else:
                    module.partition = [curr, curr+partition_size]
                    curr += partition_size
                print(f'{name} {module.partition} <- {tuple(module.weight.shape)} {CONF_DICT["granularity"]}')
                syns.append(module)
        total_size = curr
    print(f'> total_size: {total_size}')

    # >> init&load mn from ckpt
    mn = MappingNetwork(
        bit_length=BIT_LENGTH,
        output_size=total_size,
        hidden_dims=HIDDEN_DIMS,
        device=device)
    fixed_state_dict = {}
    for key, value in checkpoint['mapping_network'].items():  # Check if the key starts with 'module.' and replace it
        if key.startswith('module.'): new_key = key.replace('module.', '', 1)
        else: new_key = key
        fixed_state_dict[new_key] = value
    print(mn.load_state_dict( checkpoint['mapping_network'], strict=False,))
    mn = mn.to(device); mn.eval().requires_grad_(False)

    # >> get ex
    if EX_TYPE in ['hidden', 'jit', 'sstamp']:
        if '.torchscript.' in EX_CKPT: msg_decoder = torch.jit.load(EX_CKPT).to(device)
        else: raise NotImplementedError  # no whitening here
    elif EX_TYPE == 'random':  # current: resnet50+normal_init_fc
        from torchvision.models import resnet50, ResNet50_Weights
        msg_decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        msg_decoder.fc = torch.nn.Linear(2048,BIT_LENGTH)
        torch.nn.init.normal_(msg_decoder.fc.weight, mean=0, std=0.0275)
        # (as for bias, it is defaultly uniform inited)
        msg_decoder = msg_decoder.to(device)
    elif EX_TYPE == 'resnet':
        from torchvision.models import resnet50, ResNet50_Weights
        msg_decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        msg_decoder.fc = torch.nn.Linear(2048,32)
        msg_decoder = msg_decoder.to(device)
        msg_decoder.load_state_dict(torch.load(EX_CKPT, map_location='cpu'))
    else: raise NotImplementedError

    # >> prepare to eval
    match model_lib:
        case 'diffusers':
            vae_ori.eval()
            for param in [*vae_ori.parameters()]: param.requires_grad = False
            decoder_tune.eval().requires_grad_(False)
        case 'gan': G.eval().requires_grad_(False)
    for param in [*msg_decoder.parameters()]: param.requires_grad = False
    if 'sstamp' in EX_CKPT:  # for sstamp: make real msg_decoder func for 
        msg_decoder_model = msg_decoder
        def aux(input: torch.Tensor): # input: now in imgnet space (normalized), like -2.2~2.2
            dummy_secret = torch.zeros((input.shape[0], BIT_LENGTH), device=device)
            input_ = utils_img.unnormalize_img(input)  # make sure input is in [0, 1]
            input_ = input_.permute(0, 2, 3, 1)  # from BCHW to BHWC
            sstamp, res, message = msg_decoder_model(dummy_secret, input_)
            return message
        msg_decoder = aux
    # loader and transforms
    vqgan_transform = transforms.Compose([
        transforms.Resize(test_img_size),
        transforms.CenterCrop(test_img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan, ])
    vqgan_to_imnet = transforms.Compose(
        [utils_img.unnormalize_vqgan, utils_img.normalize_img])
    before_msg_decoder = transforms.Compose([vqgan_to_imnet])
    test_loader = utils.get_dataloader_with_caption(
        test_dir, vqgan_transform, test_batch_size,
        num_imgs=num_imgs if num_imgs > 0 else None,
        shuffle=False, num_workers=test_batch_size, collate_fn=None,
        annotations_file=anno)
    
    # >> losses
    # lossw
    if LOSSW == 'bce':
        def loss_w(decoded, keys, temp=1.0 if 'sstamp' in EX_CKPT 
            else 0.1 if EX_TYPE == 'resnet'
            else 10.0  # TODO config for this
        ): return F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')
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

    # >> rng; random generator
    if model_lib == 'diffusers':
        latent_rng = torch.Generator()
        latent_rng.manual_seed(0)
        if isinstance(pipe, DiTPipeline):
            classid_rng = torch.Generator()
            classid_rng.manual_seed(0)
            classid_set = list(set(pipe.labels.values()))
    if cli_msg == 'random':
        msg_rng = torch.Generator()
        msg_rng.manual_seed(0)
    else:
        fixed_msg = cli_msg
        msg = torch.tensor([int(k) for k in fixed_msg], dtype=TORCH_DTYPE).to(device)
        msg = msg.repeat(test_batch_size, 1)
        msg_strs = [fixed_msg]*test_batch_size

    # >> gen
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for step, (imgs_in, captions, img_ids) in enumerate(metric_logger.log_every(test_loader, 1, 'Test')):
            if cli_msg == 'random':
                msg = torch.randint(0, 2, size=(test_batch_size, BIT_LENGTH), dtype=TORCH_DTYPE, generator=msg_rng,).to(device)
                msg_strs = ["".join([str(int(ii)) for ii in msg.tolist()[jj]]) for jj in range(test_batch_size)]
            # fs: (filename, exist)s
            infs, resfs, wfs, zfs = [], [], [], []
            for msg_str, img_id in zip(msg_strs, img_ids):
                tmp = os.path.join(save_in_to, f'{img_id:06}.png')
                infs.append((tmp, os.path.exists(tmp)))
                tmp = os.path.join( CLEAN_MODEL_DIR, 'test_caption', f'{img_id:06}.png')
                resfs.append((tmp, os.path.exists(tmp)))
                tmp = os.path.join( ODIR, 'test_caption', f'{img_id:06}_{msg_str}.png')
                wfs.append((tmp, os.path.exists(tmp)))
                tmp = os.path.join( CLEAN_MODEL_DIR, 'caption_latent', f'{img_id:06}.z')
                zfs.append((tmp, os.path.exists(tmp)))

            # dont gen if existent (currently no save in)
            if save_in and (not all_exist(infs) or overwrite):
                for img_in, (inf,infe) in zip(imgs_in, infs):
                    img_in = img_in.unsqueeze(0)
                    save_image(torch.clamp(utils_img.unnormalize_vqgan(img_in), 0, 1), inf, nrow=8)
            match model_lib:
                case 'diffusers':
                    if save_z_res and (not all_exist(zfs) or overwrite):
                        if isinstance(pipe, StableDiffusionXLPipeline):
                            latents = pipe(prompt=captions, **utils.get_pipe_step_args(MODEL_ID), generator=latent_rng, output_type='latent',)[0]
                        else:  # DiT: random classid in classid_set, latent_gen as the rng, same len with captions -> List[int]
                            assert isinstance(pipe, DiTPipeline)
                            rand_indices = torch.randint(0, len(classid_set), size=(len(captions),), generator=classid_rng)
                            classids = [classid_set[idx] for idx in rand_indices.tolist()]
                            latents = pipe(class_labels=classids, **utils.get_pipe_step_args(MODEL_ID), generator=latent_rng, output_type='latent',)[0]
                        latents = latents / pipe.vae.config.scaling_factor
                        latents = pipe.vae.post_quant_conv(latents)
                        # NO Z SAVING FOR NOW
                        # for latent, (zf, zfe) in zip(latents, zfs):
                        #     torch.save(latent, zf)
                    else:  # load latent instead
                        latents = []
                        for zf, zfe in zfs: latents.append( torch.load(zf, map_location=device))
                        latents = torch.stack(latents)
                    if save_z_res and (not all_exist(resfs) or overwrite):
                        imgs_res = decoder_ori(latents)
                        for img_res, (rf, rfe) in zip(imgs_res, resfs):
                            img_res = img_res.unsqueeze(0)
                            save_image( torch.clamp(utils_img.unnormalize_vqgan(img_res), 0, 1), rf, nrow=8)
                    if save_w and (not all_exist(wfs) or overwrite):
                        flat_maps = mn(msg)
                        imgs_w = decoder_tune(latents, maps=flat_maps)
                        for img_w, (wf,wfe) in zip(imgs_w, wfs):
                            img_w = img_w.unsqueeze(0)
                            save_image( torch.clamp(utils_img.unnormalize_vqgan(img_w), 0, 1), wf, nrow=8)
                case 'gan':
                    w = gen_utils.get_w_from_seed(
                        G, test_batch_size, device, truncation_psi=truncation_psi, seed=step,
                        centroids_path=centroids_path, class_idx=class_idx)
                    if save_z_res and (not all_exist(resfs) or overwrite):
                        imgs_res = G.synthesis(w, None, **Gargs)
                        for img_res, (rf, rfe) in zip(imgs_res, resfs):
                            img_res = img_res.unsqueeze(0)
                            save_image( torch.clamp(utils_img.unnormalize_vqgan(img_res), 0, 1), rf, nrow=8)
                    if save_w and (not all_exist(wfs) or overwrite):
                        flat_maps = mn(msg)
                        imgs_w = G.synthesis(w, flat_maps, **Gargs)
                        for img_w, (wf,wfe) in zip(imgs_w, wfs):
                            img_w = img_w.unsqueeze(0)
                            save_image( torch.clamp(utils_img.unnormalize_vqgan(img_w), 0, 1), wf, nrow=8)
                case _: raise NotImplementedError
    img_dir = os.path.join(ODIR, 'test_caption')
    img_dir_nw = os.path.join(CLEAN_MODEL_DIR, 'test_caption')
    test_result_dir = os.path.join(ODIR, 'test_caption_result')
    os.makedirs(test_result_dir, exist_ok=True)
    return img_dir, img_dir_nw, test_result_dir


def save_imgs(img_dir, img_dir_nw, save_dir, num_imgs=None, mult=10):
    filenames = os.listdir(img_dir)
    # remove all filenames that are not png
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    filenames.sort()
    if num_imgs is not None: filenames = filenames[:num_imgs]
    for ii, filename in enumerate(tqdm(filenames)):
        # remove "_***" part in the stem as nw_filename
        nw_filename = Path(filename)
        nw_filename = nw_filename.with_stem(nw_filename.stem.split('_')[0])
        nw_filename = str(nw_filename)
        img_1 = Image.open(os.path.join(img_dir_nw, nw_filename))
        img_2 = Image.open(os.path.join(img_dir, filename))
        diff = np.abs(np.asarray(img_1).astype(int) - np.asarray(img_2).astype(int)) * 10
        diff = Image.fromarray(diff.astype(np.uint8))
        shutil.copy(os.path.join(img_dir_nw, nw_filename), os.path.join(save_dir, f"{ii:02d}_nw.png"))
        shutil.copy(os.path.join(img_dir, filename), os.path.join(save_dir, f"{ii:02d}_w.png"))
        diff.save(os.path.join(save_dir, f"{ii:02d}_diff.png"))

def get_img_metric(img_dir, img_dir_nw, num_imgs=None):
    filenames = os.listdir(img_dir)
    # remove all filenames that are not png
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    filenames.sort()
    if num_imgs is not None: filenames = filenames[:num_imgs]
    log_stats = []
    for ii, filename in enumerate(tqdm(filenames)):
        nw_filename = Path(filename)
        nw_filename = nw_filename.with_stem(nw_filename.stem.split('_')[0])
        nw_filename = str(nw_filename)
        pil_img_ori = Image.open(os.path.join(img_dir_nw, nw_filename))
        pil_img = Image.open(os.path.join(img_dir, filename))
        img_ori = np.asarray(pil_img_ori)
        img = np.asarray(pil_img)
        log_stat = {
            'filename': filename,
            'ssim': structural_similarity(img_ori, img, channel_axis=2),
            'psnr': peak_signal_noise_ratio(img_ori, img),
            'linf': np.amax(np.abs(img_ori.astype(int)-img.astype(int))) }
        log_stats.append(log_stat)
    return log_stats

def cached_fid(path1, path2, batch_size=32, device='cuda:0', dims=2048, num_workers=10):
    '''cached fid for path2'''
    for p in [path1, path2]:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # cache path2
    storage_path = Path.home() / f'.cache/torch/fid/{path2.replace("/", "_")}'
    if (storage_path / 'm.pt').exists():
        print(f'> Loading cached FID stats for {path2}, clean ~/.cache/torch/fid if you want to recompute')
        m2 = torch.load(storage_path / 'm.pt')
        s2 = torch.load(storage_path / 's.pt')
    else:
        storage_path.mkdir(parents=True)
        m2, s2 = compute_statistics_of_path(
            str(path2), model, batch_size, dims, device, num_workers)
        torch.save(m2, storage_path / 'm.pt')
        torch.save(s2, storage_path / 's.pt')
    m1, s1 = compute_statistics_of_path(
        str(path1), model, batch_size, dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

@torch.no_grad()
def get_bit_accs(
    img_dir: str, msg_decoder: nn.Module, key: torch.Tensor, batch_size: int = 16, attacks: dict = {}, num_imgs: int = None):
    transform = transforms.Compose([  # resize crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225]), ])
    data_loader = utils.get_dataloader(img_dir, transform, batch_size=batch_size, num_imgs=num_imgs, collate_fn=None)
    log_stats = {ii: {} for ii in range(len(data_loader.dataset))}

    STATIC_KEY = key.ndim == 1
    if STATIC_KEY: static_keys = key.repeat(batch_size, 1)
    else: key_bths = key.split(batch_size, dim=0)
    print(f'>>> STATIC_KEY: {STATIC_KEY}')

    for ii, imgs in enumerate(tqdm(data_loader)):
        imgs = imgs.to(device)
        for name, attack in attacks.items():
            imgs_aug = attack(imgs)
            decoded = msg_decoder(imgs_aug)  # b c h w -> b k
            if STATIC_KEY: keys = static_keys
            else: keys = key_bths[ii]
            # if DEBUG:  # show `decoded>0` and `keys>0` as string
            #     decoded_str = "".join([('1' if el>0 else '0') for el in decoded[0].detach()])
            #     keys_str = "".join([('1' if el else '0') for el in keys[0].detach()])
            #     print(f'>>> Decoded: {decoded_str}')
            #     print(f'>>> Keys:    {keys_str}')
            diff = (~torch.logical_xor(decoded > 0, keys > 0))  # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
            word_accs = (bit_accs == 1)  # b
            for jj in range(bit_accs.shape[0]):
                img_num = ii*batch_size+jj
                log_stat = log_stats[img_num]
                log_stat[f'bit_acc_{name}'] = bit_accs[jj].item()
                log_stat[f'word_acc_{name}'] = 1.0 if word_accs[jj].item() else 0.0
    log_stats = [{'img': img_num, **log_stats[img_num]} for img_num in range(len(data_loader.dataset))]
    return log_stats

@torch.no_grad()
def get_msgs(img_dir: str, msg_decoder: nn.Module, batch_size: int = 16, attacks: dict = {}):
    transform = transforms.Compose([transforms.ToTensor(),  # resize crop
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[ 0.229, 0.224, 0.225]), ])
    data_loader = utils.get_dataloader(img_dir, transform, batch_size=batch_size, collate_fn=None)
    log_stats = {ii: {} for ii in range(len(data_loader.dataset))}
    for ii, imgs in enumerate(tqdm(data_loader)):
        imgs = imgs.to(device)
        for name, attack in attacks.items():
            imgs_aug = attack(imgs)
            decoded = msg_decoder(imgs_aug) > 0  # b c h w -> b k
            for jj in range(decoded.shape[0]):
                img_num = ii*batch_size+jj
                log_stat = log_stats[img_num]
                log_stat[f'decoded_{name}'] = "".join([('1' if el else '0') for el in decoded[jj].detach()])
    log_stats = [{'img': img_num, **log_stats[img_num]} for img_num in range(len(data_loader.dataset))]
    return log_stats

@torch.no_grad()
def get_keys_in_filenames(img_dir: str):
    """
    Filename format: 000000_010101010(bit_length).png"
    return keys: list[str]
    """
    filenames = os.listdir(img_dir)
    filenames = [filename for filename in filenames if filename.endswith('.png')]  # remove all filenames that are not png
    filenames.sort()
    stems = [Path(filename).stem for filename in filenames]
    keys = [stem.split('_')[1] for stem in stems]
    return keys


@cli.command()
@click.option('--test_dir', default='../cache/val2014',)
@click.option('--img_dir_fid', default='../cache/val2014_512',)
@click.option('--test_img_size', default=512,)
@click.option('--batch_size', default=4,)
@click.option('--overwrite', default=False,)
def resize_dataset(test_dir, img_dir_fid, test_img_size, batch_size, overwrite,):
    '''Resize dataset (for FID test).'''
    print(f'>>> Resizing dataset...')
    print(f'>>> test_dir: {test_dir}')
    print(f'>>> img_dir_fid: {img_dir_fid}')
    print(f'>>> test_img_size: {test_img_size}')
    print(f'>>> batch_size: {batch_size}')
    print(f'>>> overwrite: {overwrite}')
    resize_dataset_func(test_dir, test_img_size, img_dir_fid, batch_size, overwrite)
def resize_dataset_func( folder_path: str, desired_size: int, output_path: str, batch_size: int, overwrite: bool):
    os.makedirs(output_path, exist_ok=True)  # list all files in the directory
    vqgan_transform = transforms.Compose([
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(), ])
    utils.seed_everything(0)  # for loader
    test_loader = utils.get_dataloader_with_caption(
        folder_path, vqgan_transform, batch_size, num_imgs=None, shuffle=False, 
        num_workers=batch_size, collate_fn=None, annotations_file = '../cache/annotations/captions_val2014.json',)
    for imgs_in, captions, img_ids in tqdm(test_loader):
        infs = []
        for img_id in img_ids:
            tmp = os.path.join( output_path, f'{img_id:06}.png')
            infs.append((tmp, os.path.exists(tmp)))
        for img_in, (inf,_) in zip(imgs_in, infs):
            if not all_exist(infs) or overwrite:
                img_in = img_in.unsqueeze(0)
                save_image( torch.clamp(img_in, 0, 1), inf, nrow=8)


@cli.command()
# major
@click.option('--ckpt', type=str, default='')
@click.option('--eval_imgs', type=bool, default=True)
@click.option('--eval_img2img', type=bool, default=True)
@click.option('--eval_bits', type=bool, default=False)
@click.option('--img_dir', type=str, default='')  # infer
@click.option('--img_dir_nw', type=str, default='')  # infer
@click.option('--img_dir_fid', type=str, default='')
@click.option('--output_dir', type=str, default='')  # infer
@click.option('--save_n_imgs', type=int, default=10)
# minor
@click.option('--decode_only', type=bool, default=False)
@click.option('--key_str', type=str, default="111010110101000001010111010011010100010000100111")
@click.option('--attack_mode', type=str, default="all")
@click.option('--dec_batch_size', type=int, default=32)
@click.option('--num_imgs', type=int, default=None)  # None for all
@click.option('--static_key', type=bool, default=False)
def test_after_gen(ckpt, output_dir, eval_imgs: bool, save_n_imgs: int, img_dir, img_dir_nw, img_dir_fid, eval_img2img: bool, eval_bits: bool, decode_only: bool, key_str, attack_mode, dec_batch_size, num_imgs, static_key):
    """FID/PSNR/SSIM (test_caption)"""
    #>> prepare
    # load ckpt and config
    checkpoint = torch.load(ckpt, map_location='cpu')
    CONF_DICT = checkpoint['params']
    EX_CKPT = CONF_DICT['ex_ckpt']
    EX_TYPE = CONF_DICT['ex_type']
    BIT_LENGTH = CONF_DICT['bit_length']
    SEED = CONF_DICT['seed']
    MODEL_ID = CONF_DICT['model_id']
    del checkpoint
    # infer dirs if None
    ODIR = os.path.dirname(ckpt)
    ARK_DIR = os.path.dirname(ODIR)
    CLEAN_MODEL_DIR = os.path.join(ARK_DIR, '_clean_', MODEL_ID.replace('/', '_'))
    if img_dir == '': img_dir = os.path.join(ODIR, 'test_caption')
    if img_dir_nw == '': img_dir_nw = os.path.join(CLEAN_MODEL_DIR, 'test_caption')
    if eval_imgs and img_dir_fid == '': raise ValueError('img_dir_fid is required')
    if output_dir == '': output_dir = os.path.join(ODIR, 'test_caption_result')
    save_img_dir = os.path.join(output_dir, 'imgs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    # prepare
    utils.seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #>> eval imgs
    if eval_imgs:
        print(f">>> Saving {save_n_imgs} diff images...")
        if save_n_imgs > 0:
            save_imgs(img_dir, img_dir_nw, save_img_dir, num_imgs=save_n_imgs)
        if eval_img2img:
            print(f'>>> Computing img-2-img stats...')
            img_metrics = get_img_metric(img_dir, img_dir_nw, num_imgs=num_imgs)
            img_df = pd.DataFrame(img_metrics)
            img_df.to_csv(os.path.join(output_dir, 'img_metrics.csv'), index=False)
            ssims = img_df['ssim'].tolist()
            psnrs = img_df['psnr'].tolist()
            linfs = img_df['linf'].tolist()
            # lpips_ds = img_df['lpips'].tolist()
            ssim_mean, ssim_std, ssim_max, ssim_min = np.mean( ssims), np.std(ssims), np.max(ssims), np.min(ssims)
            psnr_mean, psnr_std, psnr_max, psnr_min = np.mean( psnrs), np.std(psnrs), np.max(psnrs), np.min(psnrs)
            linf_mean, linf_std, linf_max, linf_min = np.mean( linfs), np.std(linfs), np.max(linfs), np.min(linfs)
            # lpips_mean, lpips_std, lpips_max, lpips_min = np.mean(lpips_ds), np.std(lpips_ds), np.max(lpips_ds), np.min(lpips_ds)
            print( f"SSIM: {ssim_mean:.4f}±{ssim_std:.4f} [{ssim_min:.4f}, {ssim_max:.4f}]")
            print( f"PSNR: {psnr_mean:.4f}±{psnr_std:.4f} [{psnr_min:.4f}, {psnr_max:.4f}]")
            print( f"Linf: {linf_mean:.4f}±{linf_std:.4f} [{linf_min:.4f}, {linf_max:.4f}]")
            # print( f"LPIPS: {lpips_mean:.4f}±{lpips_std:.4f} [{lpips_min:.4f}, {lpips_max:.4f}]")
        print(f'>>> Computing image distribution stats...')
        fid = cached_fid(img_dir, img_dir_fid)
        print(f"FID watermark : {fid:.4f}")
        fid_nw = cached_fid(img_dir_nw, img_dir_fid)
        print(f"FID vanilla   : {fid_nw:.4f}")

    if eval_bits:
        # >> get ex
        if EX_TYPE in ['hidden', 'jit', 'sstamp']:
            if '.torchscript.' in EX_CKPT: msg_decoder = torch.jit.load(EX_CKPT).to(device)
            else: raise NotImplementedError  # no whitening here
        elif EX_TYPE == 'random': # current: resnet50+normal_init_fc
            from torchvision.models import resnet50, ResNet50_Weights
            msg_decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            msg_decoder.fc = torch.nn.Linear(2048,BIT_LENGTH)
            torch.nn.init.normal_(msg_decoder.fc.weight, mean=0, std=0.0275)
            # (as for bias, it is defaultly uniform inited)
            msg_decoder = msg_decoder.to(device)
        elif EX_TYPE == 'resnet':
            from torchvision.models import resnet50, ResNet50_Weights
            msg_decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            msg_decoder.fc = torch.nn.Linear(2048,32)
            msg_decoder = msg_decoder.to(device)
            msg_decoder.load_state_dict(torch.load(EX_CKPT, map_location='cpu'))
        else: raise NotImplementedError
        msg_decoder.eval()

        if attack_mode == 'all':
            attacks = {
                'none': lambda x: x,
                'crop_05': lambda x: utils_img.center_crop(x, 0.5),
                'crop_01': lambda x: utils_img.center_crop(x, 0.1),
                'rot_25': lambda x: utils_img.rotate(x, 25),
                'rot_90': lambda x: utils_img.rotate(x, 90),
                'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
                'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
                'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
                'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
                'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
                'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
                'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
                'resize_07': lambda x: utils_img.resize(x, 0.5),
                'resize_01': lambda x: utils_img.resize(x, 0.1),
                'overlay_text': lambda x: utils_img.overlay_text(x, [76, 111, 114, 101, 109, 32, 73, 112, 115, 117, 109]),
                'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80), }
        if attack_mode == 'few':
            attacks = {
                'none': lambda x: x,
                'crop_01': lambda x: utils_img.center_crop(x, 0.1),
                'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80), }
        if decode_only:
            log_stats = get_msgs(img_dir, msg_decoder, batch_size=dec_batch_size, attacks=attacks)
        elif static_key:
            print(f'>>> Using static key: {key_str}')
            key = torch.tensor([k == '1' for k in key_str]).to(device)
            log_stats = get_bit_accs( img_dir, msg_decoder, key, batch_size=dec_batch_size, attacks=attacks)
        else:  # get key from file names
            print(f'>>> Using keys in filenames...')
            key_str_list = get_keys_in_filenames(img_dir)
            keys = [torch.tensor([k == '1' for k in key_str]).to(device) for key_str in key_str_list]
            keys = torch.stack(keys, dim=0)
            log_stats = get_bit_accs(img_dir, msg_decoder, keys, batch_size=dec_batch_size, attacks=attacks, num_imgs=num_imgs)

        print(f'>>> Saving log stats to {output_dir}...')
        df = pd.DataFrame(log_stats)
        df.to_csv(os.path.join(output_dir, 'log_stats.csv'), index=False)
        # get avg of columns
        print(df.info())
        print(df)
        print(df.mean(axis=0))


@cli.command()
@click.pass_context
def test_fidel( ctx, ):
    # ctx.invoke(gen, ckpt=ckpt, test_dir=test_dir, anno=anno, num_imgs=num_imgs, test_img_size=test_img_size, test_batch_size=test_batch_size, overwrite=overwrite, cli_msg=cli_msg, save_in=save_in, save_z_res=save_z_res, save_w=save_w, save_in_to=save_in_to)
    # ctx.invoke(test_fidel_after_gen, TODO)
    pass


@cli.command()
def test_robust():
    """Robustness"""
    pass


if __name__ == '__main__':
    cli()