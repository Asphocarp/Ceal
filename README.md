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
pip install imgui glfw pyopengl imageio-ffmpeg pyspng ftfy timm==0.4.12 ninja
```


## Starting-Over Tips

- no more `Config` dataclass / yaml config file
- any launch-possible file accepts all args from cmd arguments


## Getting Started

```shell
python scripts/train_mn.py -c 0

# test
    # Options:
    #   --ckpt TEXT
    #   --test_dir TEXT
    #   --anno TEXT
    #   --num_imgs INTEGER         -1 for all
    #   --test_img_size INTEGER
    #   --test_batch_size INTEGER
    #   --overwrite BOOLEAN
    #   --cli_msg TEXT             random for random msg for each image
    #   --save_in BOOLEAN
    #   --save_z_res BOOLEAN
    #   --save_w BOOLEAN
    #   --save_in_to TEXT
    #   --help                     Show this message and exit.
# first time
python src/test.py gen \
    --ckpt output_turbo/0106_160738/ckpt.pth \
    --test_dir ../cache/val2014 \
    --anno ../cache/annotations/captions_val2014.json \
    --num_imgs -1 \
    --test_img_size 512 \
    --test_batch_size 24 \
    --overwrite True \
    --cli_msg random \
    --save_in True \
    --save_z_res True \
    --save_w True \
    --save_in_to ../cache/val2014_512
python src/test.py gen \
    --ckpt output_turbo/0127_214457/ckpt.pth \
    --test_dir ../cache/val2014 \
    --anno ../cache/annotations/captions_val2014.json \
    --num_imgs -1 \
    --test_img_size 512 \
    --test_batch_size 16 \
    --overwrite True \
    --cli_msg random \
    --save_in False \
    --save_z_res True \
    --save_w True
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
