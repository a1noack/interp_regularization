import argparse
import numpy as np

cmd_opt = argparse.ArgumentParser(description='Argparser for training/attack.')

# dataset and model
cmd_opt.add_argument('-save_dir', type=str, default='trained_models', help='location to save/load models to/from')
cmd_opt.add_argument('-dataset', type=str, default='CIFAR-10', help='dataset being used')
cmd_opt.add_argument('-model_name', type=str, default='WRN-28-10')
cmd_opt.add_argument('-train_type', type=str, default='st', help='st = standard training, at = adversarial training, etc.')
cmd_opt.add_argument('-n_classes', type=int, default=10, help='the number of separate classes for this dataset')

# training hyperparams
cmd_opt.add_argument('-batch_size', type=int, default=128, help='number of sample per minibatch')
cmd_opt.add_argument('-n_epochs', type=int, default=200, help='number of epochs to run total')
cmd_opt.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
cmd_opt.add_argument('-lr_decay', type=str, default='cos', help='how learning rate is decayed over time')
cmd_opt.add_argument('-dropout', type=float, default=.3, help='dropout rate to use for WRN models')

cmd_opt.add_argument('-wd', type=float, default=5e-4, help='l2 weight decay factor')
cmd_opt.add_argument('-momentum', type=float, default=.9, help='momentum for SGD')
cmd_opt.add_argument('-nesterov', type=bool, default=True, help='whether nesterov momentum is used with SGD')

cmd_opt.add_argument('-gpu', type=int, default=0, help='which GPU to use')
cmd_opt.add_argument('-print_freq', type=int, default=5, help='which epochs to print training stats')
cmd_opt.add_argument('-n_seeds', type=int, default=5, help='how many models to train with the current settings')

cmd_opt.add_argument('-lambda_jr', type=float, default=4e-3, help='loss regularization weight value')
cmd_opt.add_argument('-lambda_ir', type=float, default=4e-3, help='loss regularization weight value')

# for adversary
cmd_opt.add_argument('-attack_type', type=str, default='PGD', help='which norm is used to bound adversary default l_inf')
cmd_opt.add_argument('-norm', type=str, default='inf', help='which norm is used to bound adversary default l_inf')
cmd_opt.add_argument('-epsilon', type=float, default=8/255., help='maximum magnitude adversary can perturb input by')
cmd_opt.add_argument('-step_size', type=float, default=2/255., help='step size adversary for each iteration for PGD')
cmd_opt.add_argument('-iters', type=int, default=7, help='attack iterations')
cmd_opt.add_argument('-clip_min', type=float, default=0., help='minimum value pixel can have')
cmd_opt.add_argument('-clip_max', type=float, default=1., help='maximum value pixel can have')
cmd_opt.add_argument('-targeted', type=int, default=0, help='1 => target, 0 => untargeted')

cmd_args, _ = cmd_opt.parse_known_args()