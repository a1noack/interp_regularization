#!/bin/bash

gpu=$1
permute_percent=.1

train_type=cs
lambda_cs=3.
lambda_gm=.15

# 1 => random, 2 => L2-trained eps=1.5, 3 => L2-trained eps=2.5, 
# 4 => Linf-trained eps=.3, 5 => std. train. net, 6 => L2-trained eps=1.5(smoothgrad)
# 5=> std. train simp. grad., 7 => std. train smoothgrad
interps_version=6
thresh=1 # -10 => None
grad_abs=0 # 0 => ir_utils.loss_utils.cos_sim uses raw grads instead of abs(grads)
seed=21

attack_name=Linf_strong
update_freq=1

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
    -lambda_cs=$lambda_cs  \
    -lambda_gm=$lambda_gm  \
    -permute_percent=$permute_percent  \
    -interps_version=$interps_version  \
    -thresh=$thresh  \
    -grad_abs=$grad_abs  \
    -seed=$seed  \
    -attack_name=$attack_name  \
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
    $@

# interps_version=5
# thresh=-10
# python3 train_models.py \
#     -train_type=$train_type  \
#     -lambda_cs=$lambda_cs  \
#     -lambda_gm=$lambda_gm  \
#     -permute_percent=$permute_percent  \
#     -interps_version=$interps_version  \
#     -thresh=$thresh  \
#     -grad_abs=$grad_abs  \
#     -seed=$seed  \
#     -attack_name=$attack_name  \
#     -gpu=$gpu  \
#     -update_freq=$update_freq  \
#     -save_dir=$save_dir  \
#     -dataset=$dataset  \
#     -model_name=$model_name  \
#     -lr=$lr  \
#     -batch_size=$batch_size  \
#     -n_epochs=$n_epochs  \
#     -lr_decay=$lr_decay  \
#     -dropout=$dropout  \
#     -wd=$wd  \
#     -momentum=$momentum  \
#     -nesterov=$nesterov  \
#     $@
    
# interps_version=7
# thresh=1.25
# python3 train_models.py \
#     -train_type=$train_type  \
#     -lambda_cs=$lambda_cs  \
#     -lambda_gm=$lambda_gm  \
#     -permute_percent=$permute_percent  \
#     -interps_version=$interps_version  \
#     -thresh=$thresh  \
#     -grad_abs=$grad_abs  \
#     -seed=$seed  \
#     -attack_name=$attack_name  \
#     -gpu=$gpu  \
#     -update_freq=$update_freq  \
#     -save_dir=$save_dir  \
#     -dataset=$dataset  \
#     -model_name=$model_name  \
#     -lr=$lr  \
#     -batch_size=$batch_size  \
#     -n_epochs=$n_epochs  \
#     -lr_decay=$lr_decay  \
#     -dropout=$dropout  \
#     -wd=$wd  \
#     -momentum=$momentum  \
#     -nesterov=$nesterov  \
#     $@