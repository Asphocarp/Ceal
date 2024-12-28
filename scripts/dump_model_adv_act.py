import os
import sys
import argparse


# >> Config
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='pretrained/dec_48b_whit.torchscript.pt')
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--cuda", type=str, default='3')
parser.add_argument("--skip", type=str, default='False')
parser.add_argument("--bit_length", type=int, default=5)
args, unknown = parser.parse_known_args()
MODEL = args.model
IMG_SIZE = args.img_size
CUDA = args.cuda
SKIP = args.skip
BIT_LENGTH = args.bit_length
# handler
SHORT_MODEL_ID = \
    'H' if MODEL == 'pretrained/dec_48b_whit.torchscript.pt' else \
    'S' if MODEL == 'pretrained/sstamp.torchscript.pt' else \
    'X'
# mkdir
os.makedirs('logs', exist_ok=True)


# >> loop over all split_pos
for split_pos in ['z', *[str(i) for i in range(28)]]:

    codename = f'{SHORT_MODEL_ID}{IMG_SIZE}_A{split_pos}_B{BIT_LENGTH}'
    log_filename = f'logs/{codename}.log'

    if SKIP == 'True':
        # skip tests done
        ark_dir = f'outputs'
        folders = [f for f in os.listdir(ark_dir) if os.path.isdir(os.path.join(ark_dir, f)) and codename in f]
        if folders:
            print(f"> Skip {codename} because it's already done, folder: {folders[0]}")
            continue

    cmd = f'''
    CUDA_VISIBLE_DEVICES="{CUDA}" python src/dump_adv_act.py \
        --extractor {MODEL} \
        --img_size {IMG_SIZE} \
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
        --split_pos {split_pos} \
        --bit_length {BIT_LENGTH} \
        --store False \
        --saving True \
    | tee {log_filename}
    '''
    print(cmd)
    os.system(cmd)
    print()


# TODO: cannot keyboard interrupt entire loop
# TODO once adv, save all act!