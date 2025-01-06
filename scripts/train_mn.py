import subprocess
import os
import misc

# Ensure the output directory exists
os.makedirs("output_turbo", exist_ok=True)

command = """
CUDA_VISIBLE_DEVICES=1 python src/train.py \\
    --codename "08R_Lm-ob_R8_Hnone_I0d2_Mturbo_B48_G32" \\
    --batch_size 4 \\
    --warmup_steps 0 \\
    --consi 1.8 \\
    --output_dir "output_turbo" \\
    --ex_type "hidden" \\
    --ex_ckpt "pretrained/dec_48b_whit.torchscript.pt" \\
    --acc true \\
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
    --model_id "stabilityai/sdxl-turbo" \\
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
    --hidden_dims "[]" \\
    --hidden_channels 64 \\
    --hidden_depth 8 \\
    --hidden_redundancy 1 \\
    --validate true \\
    --val_img_size 512 \\
    --val_img_num 80 \\
    --val_batch_size 4
"""

# Add logging to file
log_file = f'logs/{misc.time_str("%m%d_%H%M%S")}.log'
command = f"{command} 2>&1 | tee {log_file}"

try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit status {e.returncode}")
    raise
