OMP_NUM_THREADS=16 \
torchrun --nproc_per_node=1 src/pretrain/main.py \
    --decoder clip \
    --val_dir ../cache/val2014/ --train_dir ../cache/train2014/ --output_dir output --eval_freq 5 \
    --img_size 224 --num_bits 48 --batch_size 16 --epochs 300 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer AdamW,lr=1e-2 \
    --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 0.0 --p_res 1.0 --p_jpeg 1.0 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w_type bce --loss_margin 1 --dist False

# p_res is resized crop
# --decoder clip --val_dir ../../../cache/val2014/ --train_dir ../../../cache/train2014/ --output_dir output --eval_freq 5 --img_size 224 --num_bits 48 --batch_size 16 --epochs 300 --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 --optimizer AdamW,lr=1e-2 --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 0.0 --p_jpeg 1.0 --scaling_w 0.3 --scale_channels False --attenuation none --loss_w_type bce --loss_margin 1 --dist False