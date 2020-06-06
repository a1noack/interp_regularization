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
cmd_opt.add_argument('-update_freq', type=int, default=5, help='which epochs to consider printing training stats and saving model')
cmd_opt.add_argument('-seed', type=int, default=42, help='current seed')

cmd_opt.add_argument('-lambda_jr', type=float, default=0, help='jacobian regularization weight value')
cmd_opt.add_argument('-lambda_ir', type=float, default=0, help='interpretation regularization weight value')
cmd_opt.add_argument('-lambda_cs', type=float, default=0, help='cosine similarity regularization weight value')
cmd_opt.add_argument('-lambda_gm', type=float, default=0, help='simple gradient regularization weight value')
cmd_opt.add_argument('-lambda_db', type=float, default=0, help='double backpropagation regularization weight value')
cmd_opt.add_argument('-track_jac', type=int, default=1, help='prints out norm of input output Jacobian during training')
cmd_opt.add_argument('-permute_percent', type=float, default=0., help='the percentage of values in each target interpretation to switch around')
cmd_opt.add_argument('-interps_version', type=int, default=0, help='controls how and from where the target interps were generated')

# for adversary
cmd_opt.add_argument('-attack_type', type=str, default='PGD', help='which norm is used to bound adversary default l_inf')
cmd_opt.add_argument('-norm', type=str, default='inf', help='which norm is used to bound adversary default l_inf')
cmd_opt.add_argument('-epsilon', type=float, default=8/255., help='maximum magnitude adversary can perturb input by')
cmd_opt.add_argument('-step_size', type=float, default=2/255., help='step size adversary for each iteration for PGD')
cmd_opt.add_argument('-iters', type=int, default=7, help='attack iterations')
cmd_opt.add_argument('-clip_min', type=float, default=0., help='minimum value pixel can have')
cmd_opt.add_argument('-clip_max', type=float, default=1., help='maximum value pixel can have')
cmd_opt.add_argument('-targeted', type=int, default=0, help='1 => target, 0 => untargeted')
cmd_opt.add_argument('-n_attack', type=int, default=1, help='the number of times to attack all test set samples with each model')

cmd_opt.add_argument('-attack_name', type=str, default='L2_weak')

cmd_args, _ = cmd_opt.parse_known_args()