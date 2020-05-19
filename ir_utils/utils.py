import numpy as np
import torch
import os
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ir_utils.cmd_args import cmd_args as args
import ir_utils.simple_models as simple_models
import ir_utils.wide_resnet as wide_resnet

train_types = {'st':'standard training', 
               'at':'adversarial training', 
               'jr':'Jacobian Regularization', 
               'ir':'Interpretation Regularization',
               'cs':'Cosine Similarity Regularization',
               'db':'Double Backpropagation'}

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

def get_path(args, dir_path=False):
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
        model_path = model_path + f'model_pgd{args.norm}_eps{args.epsilon}_iters{args.iters}_{args.seed}.pt'
    elif args.train_type == 'cs':
        model_path = model_path + f'model_cs{args.lambda_cs}_gm{args.lambda_gm}_{args.seed}.pt'
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

def threshold(a, scale):
    """Used to threshold a tensor."""
    std = a.std()
    mean = a.mean()
    a = torch.where(a > mean + scale * std, a, torch.tensor([0.]))
    
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