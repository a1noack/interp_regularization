#!/bin/bash

gpu=0
train_type=$1
lambda_jr=$2
lambda_ir=$2

n_seeds=1
save_dir=trained_models
dataset=MNIST
model_name=SimpleCNN

attack_type=PGD
norm=inf
epsilon=.3
# (2.5 * epsilon) / iters
step_size=.01875
iters=40

# norm=2
# epsilon=1.5
# step_size=.32
# iters=40

clip_min=0.
clip_max=1.
targeted=0

python3 attack_models.py \
    -train_type=$train_type  \
    -lambda_jr=$lambda_jr  \
    -lambda_ir=$lambda_ir  \
    -gpu=$gpu  \
    -save_dir=$save_dir  \
    -n_seeds=$n_seeds at \
    -dataset=$dataset  \
    -model_name=$model_name  \
    -attack_type=$attack_type  \
    -norm=$norm  \
    -epsilon=$epsilon  \
    -step_size=$step_size  \
    -iters=$iters  \
    -clip_min=$clip_min  \
    -clip_max=$clip_max  \
    -targeted=$targeted  \
    $@