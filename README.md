# Ceal

## Env
Create a fresh new environment named `ceal`

```shell
# mamba env remove --name ceal
mamba create -n ceal python=3.10 --yes
mamba activate ceal
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
mamba install pillow matplotlib diffusers ipykernel transformers --yes
mamba install omegaconf accelerate deepspeed --yes
pip install augly UltraDict gradio
pip install transformers --force-reinstall
pip install lpips pycocotools pytorch_fid gradio einops wandb python-dotenv scikit-image scikit-learn 
pip install kornia flask
pip install imgui glfw pyopengl imageio-ffmpeg pyspng ftfy timm==0.4.12 ninja  # for StyleGAN
```

Datasets in `../cache/`.

```shell
# COCO [TODO]
# imagenet1k-val for fid
curl -L -o ../cache/imagenet1k-val.zip \
    https://www.kaggle.com/api/v1/datasets/download/titericz/imagenet1k-val
mkdir -p ./imagenet-val-flat && find ./imagenet-val -type f -exec mv {} ./imagenet-val-flat/ \;
```

## Starting-Over Tips

- no more `Config` dataclass / yaml config file
- any launch-possible file accepts all args from cmd arguments


## Getting Started

```shell
# train
python scripts/train_mn.py -c 0

# Turbo G-Normal-midHalf
python src/test.py gen \
    --ckpt output_turbo/0106_160738/ckpt.pth \
    --test_dir ../cache/val2014 \
    --anno ../cache/annotations/captions_val2014.json \
    --num_imgs -1 \
    --test_img_size 512 \
    --test_batch_size 24 \
    --overwrite True \
    --cli_msg random \
    --save_in False \
    --save_z_res True \
    --save_w True

# resize
CUDA_VISIBLE_DEVICES="3" python src/test.py resize-dataset --overwrite True --batch_size 16
mkdir -p ../cache/imagenet512 ../cache/imagenet768
    # 95min
CUDA_VISIBLE_DEVICES="3" python src/test.py resize-dataset \
    --overwrite True --batch_size 4 \
    --test_dir ../cache/imagenet-val-flat \
    --test_img_size 512 \
    --img_dir_fid ../cache/imagenet512
CUDA_VISIBLE_DEVICES="3" python src/test.py resize-dataset \
    --overwrite True --batch_size 4 \
    --test_dir ../cache/imagenet-val-flat \
    --test_img_size 768 \
    --img_dir_fid ../cache/imagenet768

# fidel first (infer img_dir&img_dir_nw from ckpt)
    # just fid
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0106_160738/ckpt.pth \
    --eval_imgs True --eval_img2img False --eval_bits False \
    --img_dir_fid ../cache/val2014_512 \
    --save_n_imgs 10
    # img2img 45min(for 41k image)
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0106_160738/ckpt.pth \
    --eval_imgs True --eval_img2img True --eval_bits False \
    --img_dir_fid ../cache/val2014_512 \
    --save_n_imgs 10
    # just bits 70min; B32=>13G; 6400img=>10min
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0106_160738/ckpt.pth \
    --eval_imgs False --eval_img2img False --eval_bits True \
    --num_imgs 6400 \
    --save_n_imgs 10
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0106_160738/ckpt.pth \
    --eval_imgs True --eval_img2img True --eval_bits True \
    --img_dir_fid ../cache/val2014_512 \
    --save_n_imgs 10 --num_imgs 3200

# ---
# LCM (only 768, see get_pipe_step_args; 13h)
CUDA_VISIBLE_DEVICES="1" python src/test.py gen \
    --ckpt output_turbo/0130_033407/ckpt.pth \
    --test_dir ../cache/val2014 \
    --anno ../cache/annotations/captions_val2014.json \
    --num_imgs -1 \
    --test_batch_size 4 \
    --overwrite True \
    --cli_msg random \
    --save_in False \
    --save_z_res True \
    --save_w True
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0130_033407/ckpt.pth \
    --eval_imgs True --eval_img2img True --eval_bits True \
    --img_dir_fid ../cache/val2014_768 \
    --save_n_imgs 10 --num_imgs 3200

# DiT
python src/test.py gen \
    --ckpt output_turbo/0127_082600/ckpt.pth \
    --test_dir ../cache/val2014 \
    --anno ../cache/annotations/captions_val2014.json \
    --num_imgs -1 \
    --test_img_size 512 \
    --test_batch_size 24 \
    --overwrite True \
    --cli_msg random \
    --save_in False \
    --save_z_res True \
    --save_w True
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0127_082600/ckpt.pth \
    --eval_imgs True --eval_img2img True --eval_bits True \
    --img_dir_fid ../cache/imagenet512 \
    --save_n_imgs 10 --num_imgs 3200

# StyleGAN-XL (around 3h)
CUDA_VISIBLE_DEVICES="3" python src/test.py gen \
    --ckpt output_turbo/0129_095122/ckpt.pth \
    --test_dir ../cache/val2014 \
    --anno ../cache/annotations/captions_val2014.json \
    --num_imgs -1 \
    --test_img_size 512 \
    --test_batch_size 24 \
    --overwrite True \
    --cli_msg random \
    --save_in False \
    --save_z_res True \
    --save_w True
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0129_095122/ckpt.pth \
    --eval_imgs True --eval_img2img False --eval_bits False \
    --img_dir_fid ../cache/imagenet512 \
    --save_n_imgs 10
CUDA_VISIBLE_DEVICES="3" python src/test.py test-after-gen \
    --ckpt output_turbo/0129_095122/ckpt.pth \
    --eval_imgs False --eval_img2img True --eval_bits True \
    --img_dir_fid ../cache/imagenet512 \
    --save_n_imgs 10 --num_imgs 3200
```




















## Log

- dump_model_adv_act.py: dump all adv act from a model
```shell
python scripts/dump_model_adv_act.py \
    --model pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --skip False \
    --bit_length 5 \
    --cuda 2

python scripts/dump_model_adv_act.py \
    --model pretrained/sstamp.torchscript.pt \
    --img_size 400 \
    --skip False \
    --bit_length 5 \
    --cuda 2
```

- dump_adv_act.py: dump some adv act from a layer
```shell
CUDA_VISIBLE_DEVICES="0" python src/dump_adv_act.py \
    --extractor pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --batch_size 4 \
    --num_imgs 64 \
    --msg_each 100 \
    --msg_in_order False \
    --norm_alpha 1 \
    --norm_epsilon 8 \
    --adapt_alpha_epsilon False \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos z \
    --bit_length 32 \
    --store False \
    --saving True

CUDA_VISIBLE_DEVICES="2" python src/dump_adv_act.py \
    --extractor pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --batch_size 4 \
    --num_imgs 128 \
    --msg_each 16 \
    --msg_in_order True \
    --norm_alpha 1 \
    --norm_epsilon 8 \
    --adapt_alpha_epsilon False \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos z \
    --bit_length 2 \
    --store False \
    --saving True
# --codename auto \
```

- debug:
```shell
# start server
CUDA_VISIBLE_DEVICES="3" python scripts/debug_flask.py
# send code
scripts/debug_send_buffer.py -b buf
```

- probe.py: train a probe to classify the adv act
```shell
# one
python src/probe.py --codename S400_A0_B5
# all
python scripts/train_probes.py \
    --model pretrained/sstamp.torchscript.pt \
    --img_size 400 \
    --skip False \
    --bit_length 5 \
    --data_dir output_turbo/
```


- latest:
```shell
CUDA_VISIBLE_DEVICES="0" python src/dump_adv_act.py \
    --extractor pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --batch_size 2 \
    --num_imgs 6400 \
    --msg_each 1 \
    --msg_in_order False \
    --norm_alpha 1 \
    --norm_epsilon 8 \
    --adapt_alpha_epsilon False \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos z \
    --bit_length 32 \
    --store False \
    --model_id stabilityai/sdxl-turbo \
    --saving True

CUDA_VISIBLE_DEVICES="0" python src/dump_adv_act.py \
    --extractor pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --batch_size 8 \
    --num_imgs 6400 \
    --msg_each 1 \
    --msg_in_order False \
    --norm_alpha 1 \
    --norm_epsilon 8 \
    --adapt_alpha_epsilon False \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos 29 \
    --bit_length 32 \
    --store False \
    --model_id stabilityai/sdxl-turbo \
    --saving True

# 28
CUDA_VISIBLE_DEVICES="0" python src/dump_adv_act.py \
    --extractor pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --batch_size 8 \
    --num_imgs 64000 \
    --msg_each 1 \
    --msg_in_order False \
    --norm_alpha 1 \
    --norm_epsilon 8 \
    --adapt_alpha_epsilon False \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos 28 \
    --bit_length 16 \
    --store False \
    --model_id stabilityai/sdxl-turbo \
    --saving True | tee logs/A28each1.log

CUDA_VISIBLE_DEVICES="1" python src/dump_adv_act.py \
    --extractor pretrained/dec_48b_whit.torchscript.pt \
    --img_size 256 \
    --batch_size 8 \
    --num_imgs 6400 \
    --msg_each 8 \
    --msg_in_order False \
    --norm_alpha 1 \
    --norm_epsilon 8 \
    --adapt_alpha_epsilon False \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos 29 \
    --bit_length 32 \
    --store False \
    --model_id stabilityai/sdxl-turbo \
    --saving True | tee logs/A29each8.log

python src/probe.py --codename H256_Az_B32 --bit_limit 5

python src/probe.py --codename H256_A029_B32 --bit_limit 1

python src/probe.py --codename H256_A28_B16 --bit_limit 1 --njobs 2 --mem_limit 100
```
