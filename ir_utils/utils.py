import numpy as np
import torch
import os
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ir_utils.cmd_args import cmd_args as args
import ir_utils.simple_models as simple_models
import ir_utils.wide_resnet as wide_resnet

MNIST_ATTACKERS = {
    'L2_weak': {"eps":1.5, 
                "nb_iter":40,
                "eps_iter":.50,
                "ord":2},
    'L2_strong': {"eps":2.5, 
                "nb_iter":40,
                "eps_iter":1.2,
                "ord":2},
    'L2_vstrong': {"eps":3.5, 
                "nb_iter":40,
                "eps_iter":1.75,
                "ord":2},
    'Linf_weak': {"eps":.1, 
                "nb_iter":40,
                "eps_iter":(2.5*.1)/40,
                "ord":np.inf},
    'Linf_strong': {"eps":.2, 
                "nb_iter":40,
                "eps_iter":(2.5*.2)/40,
                "ord":np.inf},
    'Linf_vstrong': {"eps":.25, 
                "nb_iter":40,
                "eps_iter":(2.5*.25)/40,
                "ord":np.inf},
    'Linf_vvstrong': {"eps":.3, 
                "nb_iter":40,
                "eps_iter":(2.5*.3)/40,
                "ord":np.inf}
}

CIFAR10_ATTACKERS = {
    'L2_weak': {"eps":80/255., 
                "nb_iter":40,
                "eps_iter":.04,
                "ord":2},
    'L2_strong': {"eps":320/255., 
                "nb_iter":40,
                "eps_iter":.16,
                "ord":2},
    'Linf_weak': {"eps":4/255., 
                "nb_iter":40,
                "eps_iter":1/255.,
                "ord":np.inf},
    'Linf_strong': {"eps":8/255., 
                "nb_iter":40,
                "eps_iter":2/255.,
                "ord":np.inf},
    'L2_weak_train': {"eps":80/255., 
                "nb_iter":7,
                "eps_iter":.1,
                "ord":2},
}

train_types = {'st':'standard training', 
               'at':'adversarial training', 
               'jr':'Jacobian Regularization', 
               'ir':'Interpretation Regularization',
               'cs':'Cosine Similarity Regularization',
               'db':'Double Backpropagation'}

def permute(target_interp, perm_percent):
    """Permutes perm_percent of the indices in target_interp.
    
    Args:
        target_interp (torch.Tensor): a single interpretation to be permuted. 
        perm_percent (float): value between zero and one that represents the probability that any of the
            any one of the C*H*W pixel values in the target interp will be swapped for another.
    """
    n_elements = target_interp.nelement()
    n_perm = int(n_elements * perm_percent)
    shape = target_interp.shape

    target_interp = target_interp.view(-1)
    idxs_to_perm = torch.randperm(n_elements)[:n_perm]
    values = target_interp[idxs_to_perm]

    permed_values = values[torch.randperm(n_perm)]

    target_interp[idxs_to_perm] = permed_values

    return target_interp.reshape(shape)

def scrape_dir(dir_path):
    """Returns a dictionary of all of the pytorch models contained in 
    the directory at dir_path, where the values
    """
    all_files = os.listdir(dir_path)
    model_files = [file for file in all_files if '.pt' in file]
    partitioned_files = {}
    for model_file in model_files:
        seedless_name = model_file[:model_file.rfind('_')]
        if seedless_name in partitioned_files:
            partitioned_files[seedless_name].append(model_file)
        else:
            partitioned_files[seedless_name] = [model_file]
            
    return partitioned_files

def zero_one_scale(a):
    return (a - a.min()) / (a.max() - a.min())

def get_path(args, attack_configs=None, dir_path=False):
    model_path = f'{args.save_dir}/{args.dataset}/{args.model_name}_{args.train_type}/'
    if dir_path:
        return model_path
    if args.train_type == 'st':
        model_path = model_path + f'model_{args.seed}.pt'
    elif args.train_type == 'ir':
        model_path = model_path + f'model_ir{args.lambda_ir}_{args.seed}.pt'
    elif args.train_type == 'jr':
        model_path = model_path + f'model_jr{args.lambda_jr}_{args.seed}.pt'
    elif args.train_type == 'at':
        ord = attack_configs['ord']
        eps = attack_configs['eps']
        nb_iter = attack_configs['nb_iter']
        model_path = model_path + f'model_pgd{ord}_eps{eps}_iters{nb_iter}_{args.seed}.pt'
    elif args.train_type == 'cs':
        model_path = model_path + f'model_cs{args.lambda_cs}_gm{args.lambda_gm}_pp{args.permute_percent}_v{args.interps_version}_{args.seed}.pt'
    elif args.train_type == 'db':
        model_path = model_path + f'model_db{args.lambda_db}_{args.seed}.pt'
    else:
        print(f'This is not a supported training type: {args.train_type}.')
        
    return model_path

def instantiate_model(args):
    if args.model_name == 'WRN-28-10':
        net = wide_resnet.Wide_ResNet(depth=28, widen_factor=10, 
                                      dropout_rate=args.dropout, 
                                      num_classes=args.n_classes)
    elif args.model_name == 'WRN-16-10':
        net = wide_resnet.Wide_ResNet(depth=16, widen_factor=10, 
                                      dropout_rate=args.dropout, 
                                      num_classes=args.n_classes)
    elif args.model_name == 'SimpleCNN':
        net = simple_models.SimpleCNN(p_drop=args.dropout)
    
    return net

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, h = -1, -1
    if n > 0:
        m = np.mean(a)
        if n > 1:
            se = scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        else:
            h = np.inf
    return m, h

def threshold(a, scale, abs=False):
    """Used to threshold a tensor."""
    std = a.std()
    mean = a.mean()
    top = torch.where(a > mean + scale * std, a, torch.tensor([0.]))
    if abs:
        a = top
    else:
        btm = torch.where(a < mean - scale * std, a, torch.tensor([0.]))
        a = top + btm
        
    return a 

def display(img, size=3, cmap='gist_heat'):
    """Displays image with dimensions (c,h,w)"""
    img = img.squeeze()
    if img.shape[0] == 3:
        img = img.permute(1,2,0)
    fig = plt.figure(figsize=(size, size), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax1 = fig.add_subplot(spec[0,0])
    im1 = ax1.imshow(img.squeeze().numpy(), cmap=cmap)
    plt.axis('on')
    
    plt.show()
    
def show(img, size=16):
    npimg = img.numpy()
    plt.figure(figsize = (size,size))
    plt.axis('off')
    plt.rcParams["axes.grid"] = False
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')