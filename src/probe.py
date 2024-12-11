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
import utils

@contextmanager
def timer(name='Task'):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f'{name} completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/current.yaml")
args, unknown = parser.parse_known_args()
utils.seed_everything(42)
torch.cuda.empty_cache()
# device = 'cuda'
output_dir = f'output_turbo/'


# %%==================== Config ====================
'''
CUDA_VISIBLE_DEVICES=3 python probe.py
'''

# codename = 'more_H256_Az_B5'
codename = 'S400_A0_B5'  # 512 points

batch_size_top = 128  # will be times of the original batch size
train_ratio = 0.8
# bit_length = 5  # None for full bit length
bit_length = 5  # None for full bit length

desired_feature_size = None  # none for ori, FIXME using Dropout now
dropout_ratio = 0.0

# %%==================== Find input folder ====================
# find the most recent folder in output_dir with this codename included
folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and codename in f]
folders.sort(reverse=True)  # Sort to get most recent first (as I named them like 20241029_123456)
if not folders:
    raise ValueError(f"No folder found containing '{codename}'")
folder_name = folders[0]
input_dir = os.path.join(output_dir, folder_name)
print(f"> Found input folder: {input_dir}")
# count the number of files in the input_dir/store
store_dir = os.path.join(input_dir, 'saving')
num_files = len(os.listdir(store_dir))
# and size of one
file_path = os.path.join(store_dir, os.listdir(store_dir)[0])
file_size = os.path.getsize(file_path)
print(f"> Found {num_files} files in {store_dir}, each {file_size/1024/1024:.2f} MB")
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
print(f"> Data Batch size: {ori_batch_size}, total data points: {num_files*ori_batch_size}")
# in_shape = tuple(one_data[train_from].shape[1:])
in_shape = tuple(one_data[train_from][0].shape[1:])
out_shape = tuple(one_data[train_to][0].shape[1:])
# out_shape = (bit_length,)
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
        # # drop last
        # self.files = os.listdir(path)[:-1]
        self.path = path

    def load_all_data(self):
        X_list = []
        y_list = []
        for idx in range(len(self.files)):
            data = torch.load(
                os.path.join(self.path, self.files[idx]),
                weights_only=False,
                map_location='cpu',
            )
            # x = data[train_from].numpy().reshape(-1, np.prod(in_shape))
            x = torch.stack(data[train_from]).numpy().reshape(-1, np.prod(in_shape))
            y = torch.stack(data[train_to]).numpy().reshape(-1, np.prod(out_shape))
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
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_ratio, random_state=42)

# base config
base_model = LogisticRegression(
    solver='lbfgs',
    # max_iter=1,
    max_iter=100000,
    # verbose=0,
    verbose=1,
    n_jobs=-1,
    penalty='l2',
    C=10.0,
    random_state=42,
)
multi_model = MultiOutputClassifier(
    base_model,
    n_jobs=-1
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
            n_jobs=-1,
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

# save the model
torch.save(model, os.path.join(input_dir, f'probe.pth'))
