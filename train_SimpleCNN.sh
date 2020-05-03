#!/bin/bash

train_type=jr
lambda_jr=.004
lambda_ir=.004
attack_type=PGD
norm=inf
epsilon=0.3
step_size=0.01
iters=40
clip_min=0.
clip_max=1.
targeted=0

gpu=1
n_seeds=3
print_freq=1

save_dir=trained_models
dataset=MNIST
model_name=SimpleCNN

lr=.01
batch_size=50
n_epochs=100
lr_decay=cos
dropout=.5
wd=.00001
momentum=.9
nesterov=False

python3 train_models.py \
    -train_type=$train_type  \
    -lambda_jr=$lambda_jr  \
    -lambda_ir=$lambda_ir  \
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
    -attack_type=$attack_type  \
    -norm=$norm  \
    -epsilon=$epsilon  \
    -step_size=$step_size  \
    -iters=$iters  \
    -clip_min=$clip_min  \
    -clip_max=$clip_max  \
    -targeted=$targeted  \
    $@