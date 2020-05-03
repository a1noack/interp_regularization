#!/bin/bash

train_type=st
gpu=1
n_seeds=3
print_freq=5

save_dir=trained_models
dataset=CIFAR-10
model_name=WRN-28-10

lr=.1
batch_size=128
n_epochs=200
lr_decay=cos
dropout=.3
wd=.0005
momentum=.9
nesterov=True

python3 train_models.py \
    -train_type=$train_type  \
    -gpu=$gpu  \
    -n_seeds=$n_seeds  \
    -print_freq=$print_freq  \
    -save_dir=$save_dir  \
    -dataset=$dataset  \
    -model_name=$model_name  \
    -lr=$lr  \
    -batch_size=$batch_size  \
    -n_epochs=$n_epochs  \
    -lr_decay=$lr_decay  \
    -dropout=$dropout  \
    -wd=$wd  \
    -momentum=$momentum  \
    -nesterov=$nesterov  \
    $@