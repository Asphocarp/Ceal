# Ceal

Ceal: Centralized Attribution of Generated Images via Adversarially Perturbing Weights

## Prepare env

1. Create a new environment named `ceal`.

```shell
# (make sure that miniconda is installed)
# conda env remove --name ceal  # remove existing env
conda env create -f scripts/env.yml
conda activate ceal

## raw cmds just for backing up
# conda create -n ceal python=3.10 --yes
# conda activate ceal
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
# conda install pillow matplotlib diffusers ipykernel transformers --yes
# conda install omegaconf accelerate deepspeed --yes
# pip install augly UltraDict gradio
# pip install transformers --force-reinstall
# pip install lpips pycocotools pytorch_fid gradio einops wandb python-dotenv scikit-image scikit-learn 
# pip install kornia flask
# pip install imgui glfw pyopengl imageio-ffmpeg pyspng ftfy timm==0.4.12 ninja  # for StyleGAN
```


2. Download datasets and generative models in `../cache/`.

```shell
cd ../cache

# download COCO in train2014/ and val2014/
mkdir -p train2014 val2014
curl -O http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d train2014
curl -O http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d val2014

# download imagenet1k-val for fid
curl -L -o imagenet1k-val.zip \
    https://www.kaggle.com/api/v1/datasets/download/titericz/imagenet1k-val
mkdir -p ./imagenet-val-flat && find ./imagenet-val -type f -exec mv {} ./imagenet-val-flat/ \;
```


Note:
- Some weights are already available with the repo, including:
    - exemplary 48-bit extractor `pretrained/dec_48b_whit.torchscript.pt`
    (The same extractor can be used for all experiments. Still, if you want to use a different extractor, e.g. one supporting more bits, you can train one according to sec(TODO) and put it in `pretrained/`.)
    - lpips-like loss functions `src/loss/losses/*.pth`

- Generative models will hopefully be automatically downloaded on-the-fly via `huggingface` (e.g. `stabilityai/sdxl-turbo`).



## Train mapping network (with default extractor)

- any launch-possible file accepts all args from cmd arguments / env variables (like WANDB_MODE, CUDA_VISIBLE_DEVICES, etc.)

```shell
# train mapping network (2 GPU Hour RTX4090) without wandb logging
# with default args: stabilityai/sdxl-turbo, 256x256, 30k steps, 48-bit extractor, etc.
# (see `scripts/train_mn.py` for all args)
WANDB_MODE=disabled \
python scripts/train_mn.py -c 0

# train mapping network (2 GPU Hour RTX4090) with wandb logging
python scripts/train_mn.py -c 0

# TODO: change model_id to test on more models (e.g. lcm-sdxl, DiT-XL-2-512, etc.)
```

## Evaluate performance
TODO
## Train custom extractor
TODO
## Train mapping network (with custom extractor)
TODO