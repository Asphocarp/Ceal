# Ceal

## env
Create a fresh new environment named `ceal`

```bash
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