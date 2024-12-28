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
pip install kornia
```


## Starting-Over Tips

- no more `Config` dataclass / yaml config file
- any launch-possible file accepts all args from cmd arguments

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
CUDA_VISIBLE_DEVICES="3" python dump_adv_act.py \
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

CUDA_VISIBLE_DEVICES="2" python dump_adv_act.py \
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

- temp to solve
```shell
CUDA_VISIBLE_DEVICES="2" python src/dump_adv_act.py \
    --extractor pretrained/sstamp.torchscript.pt \
    --img_size 400 \
    --batch_size 4 \
    --num_imgs 16 \
    --msg_each 32 \
    --msg_in_order True \
    --norm_alpha 1 \
    --norm_epsilon 40 \
    --adapt_alpha_epsilon True \
    --min_iter 1 \
    --max_iter 150 \
    --acc_thres 1 \
    --split_pos 0 \
    --bit_length 5 \
    --store False \
    --saving True \
    | tee logs/S400_A0_B5.log
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
# TODO: choose model and size
python src/probe.py
```