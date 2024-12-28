import os
import sys
import argparse


# >> Config
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='pretrained/dec_48b_whit.torchscript.pt')
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--skip", type=str, default='False')
parser.add_argument("--bit_length", type=int, default=5)
parser.add_argument("--data_dir", type=str, default='output_turbo/')
args, unknown = parser.parse_known_args()
MODEL = args.model
IMG_SIZE = args.img_size
SKIP = args.skip
BIT_LENGTH = args.bit_length
DATA_DIR = args.data_dir
# handler
SHORT_MODEL_ID = \
    'H' if MODEL == 'pretrained/dec_48b_whit.torchscript.pt' else \
    'S' if MODEL == 'pretrained/sstamp.torchscript.pt' else \
    'X'
# mkdir
os.makedirs('logs', exist_ok=True)

# >> hardcoded config
split_pos_list = [
    'z', 
    # *[str(i) for i in range(28)],
    *[str(i) for i in range(0, 3)],
    *[str(i) for i in range(4, 16)],
    *[str(i) for i in range(18, 22)],
    *[str(i) for i in range(24, 27)],
]

# >> loop over all split_pos
for split_pos in split_pos_list:

    codename = f'{SHORT_MODEL_ID}{IMG_SIZE}_A{split_pos}_B{BIT_LENGTH}'
    log_filename = f'logs/{codename}_probe.log'

    if SKIP == 'True':
        # skip tests done
        ark_dir = f'{DATA_DIR}'
        folders = [f for f in os.listdir(ark_dir) if os.path.isdir(os.path.join(ark_dir, f)) and codename in f]
        if folders:
            print(f"> Skip {codename} because it's already done, folder: {folders[0]}")
            continue

    cmd = f'''
    python src/probe.py \
        --codename {codename} \
        --train_ratio 0.8 \
        --data_dir {DATA_DIR} \
    | tee {log_filename}
    '''
    print(cmd)
    os.system(cmd)
    print()


# TODO: cannot keyboard interrupt entire loop
# TODO once adv, save all act!