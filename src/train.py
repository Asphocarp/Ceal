# %%
import os
import sys
import warnings

# autopep8: off

sys.path.append('src')
warnings.filterwarnings("ignore")

from typing import *
from config import conf
import utils_img
import utils
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
import torch
from copy import deepcopy
from loss.loss_provider import LossProvider
import lpips
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, DPMSolverMultistepScheduler, \
    StableDiffusionPipeline
from diffusers.models.vae import Decoder
from diffusers.models.autoencoder_kl import AutoencoderKL
from pathfinder import Pathfinder, Policy
import argparse
import wandb
from torch import nn
import numpy as np
from tqdm import tqdm
from mapper import MappingNetwork
import signal
from PIL import Image
from torch.optim import AdamW
import shutil
import gc

# autopep8: on


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/current.yaml")
args = parser.parse_args()
# config
conf.from_yaml(args.config)
conf.makedirs()
utils.seed_everything(conf.seed)

# versions:
# - GreedyPGD: 1. loose proj till in eps_i 2. GD with w&i  -  the default
# - +BinS: loose proj->exact proj (via BinSearch)
# - - +BOUNDARY_LEAP: loose proj->exact proj (via BinSearch), GD->BOUNDARY_LEAP w ⊥ i
# - +PLater: activate GreedyP after loss_w<0.1 for 20 steps
# - +PCGrad: GD->PCGrad
# ---
# - StochasticGreedy: proj once and go to next batch => more data, but maybe a little more stable against batch noise
# - GreedyGDP: 1. GD with w 2. loose proj till in eps_i => waste the last i grad
# - JustGD: normal gradient descent
# - JustPCGRAD: just normal GD with PCGrad

GreedyPGD = True
if GreedyPGD:
    BinS = False
    if BinS:
        BOUNDARY_LEAP = False
    # (config) None | [threshold, steps], like [0.1, 20], activate after lossw<0.1 (98%) for continuous 20 steps
    PLater = None  
    if PLater:
        # [activated, good step counter]
        PLater_state = [False, 0]  
    # details
    STOP_WHEN_ASCENT = True  # stop when lossi stop decreasing
    UNDO_ASCENT = False  # undo last step when lossi stop decreasing
    GOIN_MAX = 10  # give up going in when c>max, meaning bad lossi eval for this batch
    OPTIM_CLEAR_STATE = False  # clear state when changing direction # TODO try two optim?
StochasticGreedy = False
GreedyGDP = False
JustGD = False
JustPCGRAD = False
# assert choosing only one
assert sum([GreedyPGD, StochasticGreedy, GreedyGDP, JustGD, JustPCGRAD]) == 1

# other
SAVE_ALL_CKPTS = True
SIMPLE_MN = True
RANDOM_EXTRACTOR = False


def main(sweep=True):
    ascent_step = 0
    proj_max_step = 0

    torch.cuda.empty_cache()
    if conf.accelerate:
        from accelerate import Accelerator
        accelerator = Accelerator(log_with='wandb')
        device = accelerator.device
        accelerator.init_trackers(
            project_name="hyper-signature",
            config=conf.get_dict(),
            init_kwargs={"wandb": {"name": conf.code_name}},
        )
    else:
        device = conf.device
        wandb.init(
            project="hyper-signature",
            config=conf.get_dict(),
            name=conf.code_name,
        )
    if sweep:
        conf.sync_update(wandb.config)

    # get the name of current run (None if wandb is disabled I guess)
    if not conf.accelerate or accelerator.is_main_process:
        run_name: Optional[str] = wandb.run.name

    # %%
    # get ori vae
    pipe = utils.get_pipe(conf)
    vae_ori: AutoencoderKL = pipe.vae
    vae_ori = vae_ori.to(device)
    decoder_ori: Decoder = vae_ori.decoder

    # %%
    # copy, get path, init mappers
    decoder_tune: Decoder = deepcopy(decoder_ori)
    decoder_tune = decoder_tune.to(device)
    # with pathfinder
    po = Policy(**conf.get_dict())
    pf = Pathfinder(po)
    pf.explore(decoder_tune)
    pf.print_path()
    total_size = pf.init_model(decoder_tune)
    print(f'> total_size: {total_size}')
    if conf.accelerate:
        accelerator.log({'total_size': total_size}, step=0)
    else:
        wandb.log({'total_size': total_size}, step=0)
    mn = MappingNetwork(
        bit_length=conf.bit_length,
        total_size=total_size,
        hidden_dims=conf.hidden_dims,
        use_batchnorm=conf.use_batchnorm,
        constrainable=SIMPLE_MN,  # TODO config
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

    # %%
    # load decoder checkpoint (actually mn ckpt now)
    if conf.use_ldm_decoder_checkpoint != '':
        checkpoint = torch.load(conf.use_ldm_decoder_checkpoint,
                                map_location='cpu')
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

    # %%
    # msg_decoder

    # if True:  # resnet
    #     # maybe bad for small batch when eval?
    #     msg_decoder = WatermarkDecoder(conf.bit_length, 'resnet50')
    #     # use pretrained decoder
    #     if conf.use_msg_decoder_checkpoint is not None:
    #         checkpoint = torch.load(conf.use_msg_decoder_checkpoint,
    #                                 map_location='cpu')
    #         msg_decoder.load_state_dict(checkpoint['msg_decoder'],
    #                                     # strict=False,
    #                                     )
    #     msg_decoder = msg_decoder.to(device, conf.get_torch_dtype())

    if RANDOM_EXTRACTOR:
        print(f'>>> Using random extractor...')
        msg_decoder = utils.get_hidden_decoder(
            num_bits=conf.bit_length,
            redundancy=conf.msg_decoder_hidden_redundancy,
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
    else:  # hidden
        # jit
        if '.torchscript.' in conf.use_msg_decoder_checkpoint:
            msg_decoder = (torch.jit.load(conf.use_msg_decoder_checkpoint)
                           .to(device))
        else:
            msg_decoder = (utils.get_hidden_decoder(
                num_bits=conf.bit_length,
                redundancy=conf.msg_decoder_hidden_redundancy,
                num_blocks=conf.msg_decoder_hidden_depth,
                channels=conf.msg_decoder_hidden_channels)
                           .to(device))
            # use pretrained decoder
            ckpt = utils.get_hidden_decoder_ckpt(conf.use_msg_decoder_checkpoint)
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
                    conf.train_dir, transform, batch_size=16, collate_fn=None)
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
                conf.sync_set(
                    "use_msg_decoder_checkpoint",
                    conf.use_msg_decoder_checkpoint.replace(".pth", "_whit.torchscript.pth")
                )
                print(f'>>> Creating torchscript at {conf.use_msg_decoder_checkpoint}...')
                torch.jit.save(torchscript_m, conf.use_msg_decoder_checkpoint)

    # %%
    # prepare to train
    vae_ori.eval()
    for param in [*vae_ori.parameters()]:
        param.requires_grad = False
    decoder_tune.eval().requires_grad_(False)
    for param in [*msg_decoder.parameters()]:
        param.requires_grad = conf.train_msg_decoder

    # NOW make real msg_decoder func for sstamp
    if 'sstamp' in conf.use_msg_decoder_checkpoint:
        msg_decoder_model = msg_decoder
        def aux(input: torch.Tensor):
            # input: now in imgnet space (normalized), like -2.2~2.2
            dummy_secret = torch.zeros((input.shape[0], conf.bit_length), device=device)
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

    vqgan_transform = transforms.Compose([
        transforms.Resize(conf.img_size),
        transforms.CenterCrop(conf.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    val_vqgan_transform = transforms.Compose([
        transforms.Resize(conf.val_img_size),
        transforms.CenterCrop(conf.val_img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])

    utils.seed_everything(conf.seed)  # for loader
    if not conf.use_cached_latents:
        train_loader = utils.get_dataloader_then_repeat(
            conf.train_dir, vqgan_transform, conf.batch_size,
            num_imgs=conf.batch_size * conf.steps,
            shuffle=True, num_workers=4, collate_fn=None)
        val_loader = utils.get_dataloader(
            conf.val_dir, val_vqgan_transform, conf.val_batch_size,
            num_imgs=conf.val_img_num,
            shuffle=False, num_workers=4, collate_fn=None)
    else:
        train_loader = utils.latents_dataloader_repeat(
            utils.latents_filename(conf.model_id, conf.img_size),
            conf.batch_size,
            num_imgs=conf.batch_size * conf.steps,
            num_workers=4, collate_fn=None)
        val_loader = utils.latents_dataloader_repeat(
            utils.latents_filename(conf.model_id, conf.val_img_size)+'_val',
            conf.val_batch_size,
            num_imgs=conf.val_img_num,
            num_workers=4, collate_fn=None)
        vae_ori.encoder = None
        gc.collect()
        
    vqgan_to_imnet = transforms.Compose(
        [utils_img.unnormalize_vqgan, utils_img.normalize_img])

    if conf.distortion:
        before_msg_decoder = transforms.Compose([
            vqgan_to_imnet,
            # the following preprocessing is for robustness when training msg_decoder
            transforms.RandomErasing(),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.01), scale=(0.95, 1.0))
        ])
    else:
        before_msg_decoder = transforms.Compose([
            vqgan_to_imnet,
        ])
    
    is_sstamp = 'sstamp' in conf.use_msg_decoder_checkpoint

    # Create losses
    if conf.loss_w == 'bce':
        def loss_w(
            decoded, keys, 
            temp=1.0 if is_sstamp else 10.0
        ):
            return F.binary_cross_entropy_with_logits(
                decoded * temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    if conf.loss_i == 'mse':
        def loss_i(imgs_w, imgs):
            return torch.mean((imgs_w - imgs) ** 2)
    elif conf.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            'Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)

        def loss_i(imgs_w, imgs):
            return loss_percep(
                (1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif conf.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            'Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)

        def loss_i(imgs_w, imgs):
            return loss_percep(
                (1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif conf.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function(
            'SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)

        def loss_i(imgs_w, imgs):
            return loss_percep(
                (1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    elif conf.loss_i == 'lpips':
        lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)
        def loss_i(imgs_w, imgs):  # want [-1,1]
            return torch.mean(lpips_alex(imgs_w, imgs))
    else:
        raise NotImplementedError

    if conf.accelerate:
        decoder_tune, mn = accelerator.prepare(
            decoder_tune, mn
        )
        vae_ori, msg_decoder = accelerator.prepare(
            vae_ori, msg_decoder
        )

    # optimizer
    if conf.train_msg_decoder:
        to_train = all_mapper_params + list(msg_decoder.parameters())
    else:
        to_train = all_mapper_params
    optim_params = utils.parse_params(conf.optimizer)
    optimizer = utils.build_optimizer(
        model_params=to_train,
        **optim_params
    )
    def rebuild():
        op = utils.build_optimizer(
            model_params=to_train,
            **optim_params
        )
        return op

    pack_optim = optimizer
    if JustPCGRAD:
        assert not conf.accelerate
        from pcgrad import PCGrad
        pack_optim = PCGrad(optimizer)
    if GreedyPGD and (BinS or UNDO_ASCENT or OPTIM_CLEAR_STATE):
        assert not conf.accelerate
        from pcgrad import LeapGrad
        pack_optim = LeapGrad(
            optimizer,
            reduction='mean',
            rebuild=rebuild,
        )

    torch.cuda.empty_cache()

    if conf.accelerate:
        train_loader, optimizer = accelerator.prepare(
            train_loader, optimizer
        )

    # Train
    header = 'train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    base_lr = optimizer.param_groups[0]["lr"]
    if conf.accelerate:
        base_lr *= accelerator.num_processes
    no_w_step = 0

    for step, imgs_in in enumerate(metric_logger.log_every(train_loader, conf.log_freq, header)):
        if conf.use_cached_latents:
            imgs_in = imgs_in[0]  # no label
        imgs_in = imgs_in.to(device, non_blocking=True)
        imgs_in = imgs_in.type(conf.get_torch_dtype())
        utils.adjust_learning_rate(
            optimizer, step, conf.steps, conf.warmup_steps, base_lr, cosine_lr=conf.cosine_lr)
        pack_optim.zero_grad()
        # torch.cuda.empty_cache()  # waste time?

        # gen msg
        msg = torch.randint(0, 2, size=(conf.batch_size, conf.bit_length),
                            dtype=conf.get_torch_dtype(),
                            ).to(device, non_blocking=True)
        msg_str = ["".join([str(int(ii)) for ii in msg.tolist()[jj]]) for jj in range(conf.batch_size)]

        # gen image
        if not conf.use_cached_latents:
            z = vae_ori.pass_post_quant_conv(imgs_in)
        else:
            z = imgs_in
        imgs_res = vae_ori.decoder(z)  # TODO load cached res as well
        flat_maps = mn(msg)
        imgs_w = decoder_tune(z, maps=flat_maps)
        imgs_compare = imgs_res
        lossi = loss_i(imgs_w, imgs_compare)

        to_project = conf.i_constraint > 0 and lossi > conf.i_constraint

        # if StochasticGreedy:
        #     # TODO get lossw  # get lossw just for record
        #     if to_project:
        #         lossw = lossw.detach()
        #         no_w_step += 1
        #     # TODO backward, step
        
        assert GreedyPGD
        if to_project:
            goin_counter = 0  # for GOIN_MAX
            last_lossi = lossi  # for STOP_WHEN_ASCENT
            if OPTIM_CLEAR_STATE:
                optimizer = pack_optim.clear_state()
            while lossi > conf.i_constraint and (not STOP_WHEN_ASCENT or last_lossi>=lossi) and goin_counter < GOIN_MAX:
                # break when 1. done descent 2. ascent (maybe tolerance?) 3. max iter
                last_lossi = lossi
                if UNDO_ASCENT:
                    pack_optim.store_p()
                # > op.apply lossi
                (conf.lambda_i*lossi).backward(retain_graph=False)
                pack_optim.step()
                pack_optim.zero_grad()
                # > get lossi
                flat_maps = mn(msg)
                imgs_w = decoder_tune(z, maps=flat_maps)
                lossi = loss_i(imgs_w, imgs_compare)
                print(f'> going in => lossi: {lossi}')
                goin_counter += 1
                no_w_step += 1  # acctually accumulated w step (on same batch)
            # for case 2&3
            if lossi > conf.i_constraint:
                # case 2: ascent
                if last_lossi < lossi:
                    ascent_step += 1
                    if UNDO_ASCENT:
                        pack_optim.restore_p()
                        print(f'> given up for increasing, undo lossi: {last_lossi} <- {lossi}')
                        lossi = last_lossi
                        flat_maps = mn(msg)
                        imgs_w = decoder_tune(z, maps=flat_maps)
                    else:
                        print(f'> given up for increasing lossi: {last_lossi} -> {lossi}')
                    del last_lossi
                    with torch.no_grad():  # just for logging w
                        decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                        lossw = loss_w(decoded, msg)
                        loss = conf.lambda_w * lossw + conf.lambda_i * lossi
                # case 3: max iter
                elif goin_counter >= GOIN_MAX:
                    proj_max_step += 1
                    print(f'> given up for MAX ITER {goin_counter}, lossi: {lossi}')
                    with torch.no_grad():  # just for logging w
                        decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                        lossw = loss_w(decoded, msg)
                        loss = conf.lambda_w * lossw + conf.lambda_i * lossi
                if BinS:  # > release accu ag etc here (pretty old code)
                    pack_optim.release_delta()
                    pack_optim.reset_leap()
                pack_optim.zero_grad()  # for sure
            # for case 1: done proj
            else: 
                if not BinS:
                    print(f'> loose in after {goin_counter}, lossi: {lossi}')
                    # go one w step after going in (since we got i, getting w is cheap)
                    decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                    lossw = loss_w(decoded, msg)
                    loss = conf.lambda_w * lossw + conf.lambda_i * lossi
                    loss.backward()
                    pack_optim.step()
                    pack_optim.zero_grad()  # for sure
                else:  # > FOLD BinS
                    with torch.no_grad():
                        # back to boundary
                        pack_optim.set_delta()
                        al, ar, am = 0., 1., 1.
                        print(f'> al: {al:.3f}, ar: {ar:.3f}, am: {am:.3f} => lossi: {lossi}')
                        ieps = 0.01
                        # while abs(lossi-conf.i_constraint) > ieps:  # both side
                        while not (conf.i_constraint-ieps <= lossi <= conf.i_constraint):  # inside
                            am = (al+ar)/2
                            pack_optim.reapply(am)
                            # get lossi
                            flat_maps = mn(msg)
                            imgs_w = decoder_tune(z, maps=flat_maps)
                            lossi = loss_i(imgs_w, imgs_compare)
                            print(f'> al: {al:.3f}, ar: {ar:.3f}, am: {am:.3f} => lossi: {lossi}')
                            if lossi < conf.i_constraint:
                                ar = am
                            else:
                                al = am
                        pack_optim.release_delta()
                    pack_optim.zero_grad()  # for sure
                    # get lossi, lossw (at boundary)
                    flat_maps = mn(msg)
                    imgs_w = decoder_tune(z, maps=flat_maps)
                    lossi = loss_i(imgs_w, imgs_compare)
                    decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
                    lossw = loss_w(decoded, msg)
                    if BOUNDARY_LEAP:
                        # op.accu lossw
                        lossw.backward(retain_graph=True)
                        pack_optim.accumulate()
                        # op.grad_as_nv(loss_i)
                        # # just a direction, no need to mul lambda_i here
                        lossi.backward(retain_graph=False)
                        pack_optim.grad_as_nv()
                        # tangent leap
                        if OPTIM_CLEAR_STATE:
                            optimizer = pack_optim.clear_state()
                        print(f'> leap {pack_optim.counter} by {pack_optim._reduction}')
                        pack_optim.boundary_leap()
                        if OPTIM_CLEAR_STATE:
                            optimizer = pack_optim.clear_state()
                    else:
                        loss = conf.lambda_w * lossw + conf.lambda_i * lossi
                        loss.backward()
                        pack_optim.step()
                    pack_optim.zero_grad()  # for sure
        else:  # normally, no proj
            decoded = msg_decoder(before_msg_decoder(imgs_w))  # b c h w -> b k
            lossw = loss_w(decoded, msg)
            loss = conf.lambda_w * lossw + conf.lambda_i * lossi
            loss.backward()
            pack_optim.step()

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

        if conf.accelerate:
            accelerator.log(log_stats, step=step * accelerator.num_processes)
        else:
            wandb.log(log_stats, step=step)

        # save images during training
        if (step + 1) % conf.save_img_freq == 0:
            if not conf.accelerate or accelerator.is_main_process:
                if not conf.use_cached_latents:
                    save_image(
                        torch.clamp(utils_img.unnormalize_vqgan(imgs_in), 0, 1),
                        os.path.join(
                            conf.get_output_dir(),
                            'train',
                            f'{step:05}_{header}_orig.png'),
                        nrow=8)
                save_image(
                    torch.clamp(utils_img.unnormalize_vqgan(imgs_res), 0, 1),
                    os.path.join(
                        conf.get_output_dir(),
                        'train',
                        f'{step:05}_{header}_res.png'),
                    nrow=8)
                save_image(
                    torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                    os.path.join(
                        conf.get_output_dir(),
                        'train',
                        f'{step:05}_{header}_w.png'),
                    nrow=8)

        # TODO remove all intermediate objects?
        del imgs_in, z, imgs_res, flat_maps, imgs_w, imgs_compare, lossi, decoded, lossw, loss, diff1, bit_accs, word_accs

        # validate and save checkpoint
        if (step + 1) % conf.save_checkpoint_freq == 0:  # or step == 0
            # validate
            if conf.validate:
                decoder_tune.eval()
                val_stats = val(
                    device, vae_ori, decoder_tune, msg_decoder, val_loader,
                    vqgan_to_imnet, mn, step)
                if conf.accelerate:
                    accelerator.log(val_stats, step=step * accelerator.num_processes)
                else:
                    wandb.log(val_stats, step=step)
                decoder_tune.train()
            # save the latest checkpoint
            if not conf.accelerate or accelerator.is_main_process:
                if conf.accelerate:
                    dict1 = {
                        # 'decoder_tune': decoder_tune.state_dict(),
                        # 'msg_decoder': msg_decoder.state_dict(),
                        'mapping_network': accelerator.unwrap_model(mn).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'params': dict(conf.get_dict()),
                    }
                    if SAVE_ALL_CKPTS:
                        accelerator.save(dict1, os.path.join(
                            conf.get_output_dir(), f"checkpoint_{step + 1}.pth"))
                    else:
                        accelerator.save(dict1, os.path.join(
                            conf.get_output_dir(), f"checkpoint.pth"))
                else:
                    dict1 = {
                        # 'decoder_tune': decoder_tune.state_dict(),
                        # 'msg_decoder': msg_decoder.state_dict(),
                        'mapping_network': mn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'params': dict(conf.get_dict()),
                    }
                    if SAVE_ALL_CKPTS:
                        torch.save(dict1, os.path.join(
                            conf.get_output_dir(), f"checkpoint_{step + 1}.pth"))
                    else:
                        torch.save(dict1, os.path.join(
                            conf.get_output_dir(), f"checkpoint.pth"))

    print("Averaged {} stats:".format('train'), metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if conf.accelerate:
        accelerator.end_training()
    torch.cuda.empty_cache()


@torch.no_grad()
def val(device, vae_ori, decoder_tune, msg_decoder, val_loader, vqgan_to_imnet, mn, train_step):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    # for sstamp:
    sstamp_resize = transforms.Compose([
        transforms.Resize(conf.img_size),
    ])
    for ii, imgs_in in enumerate(metric_logger.log_every(val_loader, conf.log_freq, header)):
        if conf.use_cached_latents:
            imgs_in = imgs_in[0]  # no label
        imgs_in = imgs_in.to(device, non_blocking=True)
        imgs_in = imgs_in.type(conf.get_torch_dtype())
        # gen msg
        msg_val = torch.randint(0, 2, size=(conf.val_batch_size, conf.bit_length),
                                dtype=conf.get_torch_dtype(),
                                ).to(device, non_blocking=True)
        # msg_val_str = ["".join([str(int(ii)) for ii in msg_val.tolist()[jj]]) for jj in range(conf.val_batch_size)]

        # gen image
        if not conf.use_cached_latents:
            z = vae_ori.pass_post_quant_conv(imgs_in)
        else:
            z = imgs_in
        imgs_res = vae_ori.decoder(z)
        flat_maps = mn(msg_val)
        imgs_w = decoder_tune(z, maps=flat_maps)

        log_stats = {
            "psnr_val": utils_img.psnr(imgs_w, imgs_res).mean().item(),
            # "psnr_ori_val": utils_img.psnr(imgs_w, imgs).mean().item(),
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
            if 'sstamp' in conf.use_msg_decoder_checkpoint:
                imgs_aug = sstamp_resize(imgs_aug)
            decoded = msg_decoder(imgs_aug)  # b c h w -> b k
            diff = (~torch.logical_xor(decoded > 0, msg_val > 0))  # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
            word_accs = (bit_accs == 1)  # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            log_stats[f'word_acc_{name}'] = torch.mean(
                word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name: loss})

        # mkdir
        os.makedirs(
            os.path.join(
                conf.get_output_dir(),
                'validate',
                f'{train_step}'),
            exist_ok=True
        )

        if ii % conf.val_save_img_freq == 0:
            orig_filename = os.path.join(
                conf.get_output_dir(),
                'validate',
                f'{train_step}',
                f'{ii:05}_val_orig.png')
            res_filename = os.path.join(
                conf.get_output_dir(),
                'validate',
                f'{train_step}',
                f'{ii:05}_val_res.png')
            w_filename = os.path.join(
                conf.get_output_dir(),
                'validate',
                f'{train_step}',
                f'{ii:05}_val_w.png')
            diff_filename = os.path.join(
                conf.get_output_dir(),
                'validate',
                f'{train_step}',
                f'{ii:05}_val_zdiff.png')
            if not conf.use_cached_latents:
                save_image(
                    torch.clamp(utils_img.unnormalize_vqgan(imgs_in), 0, 1),
                    orig_filename,
                    nrow=8)
            save_image(
                torch.clamp(utils_img.unnormalize_vqgan(imgs_res), 0, 1),
                res_filename,
                nrow=8)
            save_image(
                torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                w_filename,
                nrow=8)
            # save the diff*10 between res and w
            # TODO optimize
            try:
                img_1 = Image.open(res_filename)
                img_2 = Image.open(w_filename)
                diff = np.abs(np.asarray(img_1).astype(int) -
                              np.asarray(img_2).astype(int)) * 10
                diff = Image.fromarray(diff.astype(np.uint8))
                diff.save(diff_filename)
            except:
                pass

    print("Averaged {} stats:".format('eval'), metric_logger)
    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return val_stats


# __main__
sweep_id = os.getenv('SWEEP_ID')
sweep_count = int(os.getenv('SWEEP_COUNT', 1))
if sweep_id:
    wandb.agent(sweep_id,
                count=sweep_count,
                function=main)
else:
    wandb.login()
    main(sweep=False)

