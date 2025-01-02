# %%
import os
from typing import *
import torch
import argparse
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from contextlib import contextmanager
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import utils
from utils import str2bool

@contextmanager
def timer(name='Task'):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f'{name} completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')


# %%==================== Config ====================
parser = argparse.ArgumentParser()
parser.add_argument("--codename", type=str, default="S400_A0_B5")
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--data_dir", type=str, default="output_turbo/")
parser.add_argument("--bit_limit", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--drop_last_file", type=str2bool, default=False)
parser.add_argument("--mem_limit", type=int, default=None, help="as GB currently, None for no limit")
parser.add_argument("--njobs", type=int, default=8)
args, unknown = parser.parse_known_args()

# >> rename
DATA_DIR = args.data_dir
CODENAME = args.codename
TRAIN_RATIO = args.train_ratio
BIT_LIMIT = args.bit_limit
SEED = args.seed
DROP_LAST_FILE = args.drop_last_file
NJOBS = args.njobs
MEM_LIMIT = args.mem_limit

# >> handler
desired_feature_size = None  # deprecated
dropout_ratio = 0.0  # deprecated
utils.seed_everything(SEED)
torch.cuda.empty_cache()


# %%==================== Find input folder ====================
# find the most recent folder in DATA_DIR with this codename included
folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and CODENAME in f]
folders.sort(reverse=True)  # Sort to get most recent first (as I named them like 20241029_123456)
if not folders:
    raise ValueError(f"No folder found containing '{CODENAME}'")
folder_name = folders[0]
input_dir = os.path.join(DATA_DIR, folder_name)
print(f"> Found input folder: {input_dir}")
# count the number of files in the input_dir/store
store_dir = os.path.join(input_dir, 'saving')
num_files = len(os.listdir(store_dir))
# and size of one
file_path = os.path.join(store_dir, os.listdir(store_dir)[0])
file_size = os.path.getsize(file_path)
print(f"> Found {num_files} files in {store_dir}, each {file_size/1024/1024:.2f} MB")
# limit mem usage
if MEM_LIMIT is not None:
    file_to_load_num = MEM_LIMIT * 1024 * 1024 * 1024 // file_size
    file_to_load_num = min(file_to_load_num, num_files)
    print(f"> Memory limit: {MEM_LIMIT} GB, Loading {file_to_load_num} files")
else:
    file_to_load_num = num_files
    print(f"> No Memory limit, Loading {file_to_load_num} files")
# load one file via torch, show the dict keys
one_data = torch.load(file_path, map_location='cpu', weights_only=False)
print(f"> Data keys: {list(one_data.keys())}")
# we train from key 'act_adv' to 'decoded'
# train_to = 'decoded'
# train_from = 'act_adv'
# train_from = 'act'
# train_to = 'act_adv'
train_from = 'act_adv'
train_to = 'msg'
# infer the model I/O size (ignore ori_batch_size)
ori_batch_size = one_data[train_to][0].shape[0] * len(one_data[train_to])
print(f"> Data Batch size: {ori_batch_size}, total data points: {num_files*ori_batch_size}, loaded data points: {file_to_load_num*ori_batch_size}")
# in_shape = tuple(one_data[train_from].shape[1:])
in_shape = tuple(one_data[train_from][0].shape[1:])
ori_out_shape = tuple(one_data[train_to][0].shape[1:])
out_shape = (BIT_LIMIT,)
print(f"> Model I/O: {train_from} -> {train_to}; {list(in_shape)} -> {list(out_shape)}; {np.prod(in_shape)} -> {np.prod(out_shape)}")

in_size = desired_feature_size if desired_feature_size is not None else np.prod(in_shape)
sub_feature_ratio = in_size / np.prod(in_shape)
print(f"> Sub feature ratio: {sub_feature_ratio}, Dropout ratio: {dropout_ratio}")
out_size = np.prod(out_shape)
if desired_feature_size is not None:
    sub_feature_set = torch.randperm(np.prod(in_shape))[:in_size]
else:
    sub_feature_set = slice(None)

# infer the parameter size (one linear layer of MLP), and VRAM usage
# param_size = in_size * np.prod(out_shape)
# rank = 512
# param_size = in_size * rank + rank * out_size * (bit_length+1)
# param_size = in_size * out_size
param_size = (in_size+1) * out_size
vram_usage = param_size * 4 / 1024 / 1024 / 1024 # GB
print(f"> Parameter size: {param_size}={param_size/1e9:.2f}B params; VRAM usage: {vram_usage:.2f} GB")

# %%
# make a pytorch dataloader out of IO data, dont load all data right now
# ! maybe break the batches of the original folder, so that we can load randomly
# ! (now require new batch to be times of the original batch size, break them later)
class IODataset:
    """A dataset class for loading data and returning features and labels in numpy array format."""
    def __init__(self, path):
        self.files = os.listdir(path)
        if DROP_LAST_FILE:
            self.files = self.files[:-1]
        self.files = self.files[:file_to_load_num]
        self.path = path

    def load_all_data(self):
        X_list = []
        y_list = []
        for idx in tqdm(range(len(self.files)), desc='Loading files', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            data = torch.load(
                os.path.join(self.path, self.files[idx]),
                weights_only=False,
                map_location='cpu',
            )
            # x = data[train_from].numpy().reshape(-1, np.prod(in_shape))
            x = torch.stack(data[train_from]).numpy().reshape(-1, np.prod(in_shape))
            y = torch.stack(data[train_to]).numpy().reshape(-1, np.prod(ori_out_shape))
            y = y[:, :BIT_LIMIT]
            y = (y > 0.5).astype(int)  # Binarize the labels (float to int)
            X_list.append(x)
            y_list.append(y)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

# Load data
print('Loading data...')
with timer('Loading data'):
    dataset = IODataset(store_dir)
    X, y = dataset.load_all_data()

# Check if the number of samples in X and y are consistent
print(f"Total data samples: X.shape = {X.shape}, y.shape = {y.shape}")

# # If data is too large, optionally sample a subset
# max_samples = 100000
# if X.shape[0] > max_samples:
#     idx = np.random.choice(X.shape[0], max_samples, replace=False)
#     X = X[idx]
#     y = y[idx]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=SEED)

# base config
base_model = LogisticRegression(
    solver='lbfgs',
    # max_iter=1,
    max_iter=100000,
    # verbose=0,
    verbose=1,
    n_jobs=NJOBS,
    penalty='l2',
    C=10.0,
    random_state=SEED,
)
multi_model = MultiOutputClassifier(
    base_model,
    n_jobs=NJOBS
)

with timer('Training'):
    GRID_SEARCH = False
    if GRID_SEARCH:
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__penalty': ['l2']
        }
        # grid search with cross-validation
        print('Training model with cross-validation...')
        grid_search = GridSearchCV(
            multi_model,
            param_grid,
            cv=5,
            n_jobs=NJOBS,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = multi_model
        model.fit(X_train, y_train)

# evaluate
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

train_score = accuracy_score(y_train, y_train_pred)
val_score = accuracy_score(y_val, y_val_pred)
word_train_score = (y_train == y_train_pred).all(axis=1).mean()
word_val_score = (y_val == y_val_pred).all(axis=1).mean()

print(f'Bit Train/Val accuracy: {train_score}, {val_score}')
print(f'Word Train/Val accuracy: {word_train_score}, {word_val_score}')

avg_train = train_score ** (1/BIT_LIMIT)
avg_val = val_score ** (1/BIT_LIMIT)
print(f'Avg Train/Val Accuracy per Bit: {avg_train}, {avg_val}')

# save the model
torch.save(model, os.path.join(input_dir, f'probe{BIT_LIMIT}.pth'))
