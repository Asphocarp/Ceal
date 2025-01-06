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


print(f'> Override args: {override}')

    # --codename "08R_Lm-ob_R8_Hnone_I0d2_Mturbo_B48_G32" \\
command = f"""
CUDA_VISIBLE_DEVICES={CUDA} python src/train.py \\
    --batch_size 4 \\
    --consi 1.8 \\
    --model_id "stabilityai/sdxl-turbo" \\
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
    --steps 100000 \\
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
    --layer_end "up_blocks.3.resnets.0.conv1" \\
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
    --conv_out_full_out false \\
    --conv_in_null_in true \\
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
