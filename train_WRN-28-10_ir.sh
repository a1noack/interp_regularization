#!/bin/bash

gpu=2

train_type=cs
lambda_cs=.75
lambda_gm=.005
permute_percent=0
# 0 => pgdL2_eps1.2549_iters7_smoothgrad, 1 => pgdL2_eps0.314_iters7_simp_unproc
# 2 => pgdL2_eps0.314_iters7_smooth_unproc, 3 => std_train_simp_unproc, 4 => std_train_smooth_unproc
interps_version=4
thresh=1 # -10 => None
grad_abs=0 # 0 => ir_utils.loss_utils.cos_sim uses raw grads instead of abs(grads)
seed=99

attack_name=L2_weak

update_freq=1

save_dir=trained_models
dataset=CIFAR-10
model_name=WRN-28-10

lr=.1 # divide by 8 for JR, usually .1
batch_size=128 # divide by 8 for JR, usually 128
n_epochs=200
lr_decay=cos
dropout=.3
wd=.0005
momentum=.9
nesterov=True

python train_models.py \
    -train_type=$train_type  \
    -lambda_cs=$lambda_cs  \
    -lambda_gm=$lambda_gm  \
    -permute_percent=$permute_percent  \
    -interps_version=$interps_version  \
    -thresh=$thresh  \
    -grad_abs=$grad_abs  \
    -gpu=$gpu  \
    -update_freq=$update_freq  \
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
    -seed=$seed  \
    $@