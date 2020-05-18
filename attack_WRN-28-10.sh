#!/bin/bash

gpu=0
train_type=$1

seed=42
n_attack=3 # the number of times to repeat attack
save_dir=trained_models
dataset=CIFAR-10
model_name=WRN-28-10

attack_type=PGD

# Linf settings
norm=inf
iters=40
# large epsilon
# epsilon=0.0314 # = 8/255
# step_size=0.00784 # = 2/255
# small epsilon
epsilon=0.015686 # = 4/255
step_size=0.0039215 # = 1/255

# L2 adversary settings
# norm=2
# iters=40
# large epsilon
# epsilon=1.2549 # = at320/255
# step_size=.16
# small epsilon
# epsilon=.314 # = 80/255
# step_size=.04


clip_min=0.
clip_max=1.
targeted=0

python3 attack_models.py \
    -train_type=$train_type  \
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
    -n_attack=$n_attack  \
    $@