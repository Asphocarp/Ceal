import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import subprocess
import misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cuda", type=str, default="2")
args, override = parser.parse_known_args()

CUDA = args.cuda
os.makedirs("output_turbo", exist_ok=True)


'''
EGs: !gan needs nvcc to compile some pytorch ops!
# Debug
python scripts/train_mn.py -c 1 --debug true
# Normal
python scripts/train_mn.py -c 3
# ResRescaled 
python scripts/train_mn.py -c 1 \
    --bit_length 32 --ex_type "resnet" --ex_ckpt "pretrained/resnet.pth" --consi 2.5
# ResReLd025
python scripts/train_mn.py -c 1 \
    --bit_length 32 --ex_type "resnet" --ex_ckpt "pretrained/resnet.pth" --consi 2.5 \
    --lambdai 0.025
# ResReLd02More; but improve metrics later
python scripts/train_mn.py -c 1 \
    --bit_length 32 --ex_type "resnet" --ex_ckpt "pretrained/resnet.pth" --consi 2.5 \
    --lambdai 0.02 --layer_end "up_blocks.3.resnets.0.conv1"
# Random
python scripts/train_mn.py -c 1 \
    --bit_length 32 --ex_type "random" --consi 2.5

    # old
    --output_dir "output_turbo" \\
    --layer_end "up_blocks.3.resnets.0.conv1" \\
    --conv_out_full_out false \\
    --steps 100000 \\
    --consi 1.8 \\
# Normal2.5
python scripts/train_mn.py -c 0
# Normal1 - 3 (8*0.25 steps)
python scripts/train_mn.py -c 1 \
    --consi 1.0
python scripts/train_mn.py -c 1 \
    --consi 1.25
python scripts/train_mn.py -c 1 \
    --consi 1.5
python scripts/train_mn.py -c 1 \
    --consi 1.75
# lcm C2.5
python scripts/train_mn.py -c 3 \
    --model_id "latent-consistency/lcm-sdxl" --consi 1.75
# DiT C2.5
python scripts/train_mn.py -c 3 \
    --model_id "../cache/DiT-XL-2-512"
# just except last conv
python scripts/train_mn.py -c 0 --conv_out_full_out false --layer_end up_blocks.3.resnets.2.conv2
# gan-e1
python scripts/train_mn.py -c 1 \
    --model_id "stylegan-xl:https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet512.pkl"
# gan-e2 include bias, consi3.5
python scripts/train_mn.py -c 3 \
    --model_id "stylegan-xl:https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet512.pkl" \
    --include_bias true --consi 3.25
'''
print(f'> override args: {override}')

command = f"""
CUDA_VISIBLE_DEVICES={CUDA} python src/train_gan.py \\
    --batch_size 4 \\
    --consi 1.75 \\
    --model_id "../cache/sdxl-turbo" \\
    --output_dir "output_turbo" \\
    --warmup_steps 0 \\
    --ex_type "hidden" \\
    --ex_ckpt "pretrained/dec_48b_whit.torchscript.pt" \\
    --acc false \\
    --save_all_ckpts false \\
    --seed 0 \\
    --train_dir "../cache/train2014/" \\
    --val_dir "../cache/val2014/" \\
    --torch_dtype_str "float32" \\
    --img_size 256 \\
    --steps 30000 \\
    --optimizer "AdamW,lr=1e-4" \\
    --cosine_lr false \\
    --bit_length 48 \\
    --lossw "bce" \\
    --lossi "watson-vgg" \\
    --lambdai 0.05 \\
    --train_ex false \\
    --distortion false \\
    --log_freq 10 \\
    --save_img_freq 50 \\
    --save_ckpt_freq 2000 \\
    --local_files_only true \\
    --use_safetensors true \\
    --use_cached_latents false \\
    --granularity "kernel" \\
    --layer_selection "layer_range" \\
    --layer_begin "up_blocks.1.resnets.0.conv1" \\
    --layer_end "conv_out" \\
    --conv_out_full_out true \\
    --conv_in_null_in true \\
    --use_lora true \\
    --lora_rank 8 \\
    --channel_selection "random" \\
    --include_bias false \\
    --enable_group true \\
    --continuous_groups true \\
    --chain true \\
    --total_group_num 32 \\
    --group_num 32 \\
    --start_group 11 \\
    --absolute_perturb false \\
    --hidden_dims "[1024]" \\
    --hidden_channels 64 \\
    --hidden_depth 8 \\
    --hidden_redundancy 1 \\
    --validate true \\
    --val_img_size 512 \\
    --val_img_num 80 \\
    --val_batch_size 4 \\
    {' '.join(override)}
"""

# Add logging to file
log_file = f'logs/{misc.time_str("%m%d_%H%M%S")}.log'
command = f"{command} 2>&1 | tee {log_file}"

try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit status {e.returncode}")
    raise
