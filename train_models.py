import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from advertorch.attacks import PGDAttack
import ir_utils.resnet as resnet
import ir_utils.simple_models as simple_models
import ir_utils.wide_resnet as wide_resnet
from ir_utils.cmd_args import cmd_args as args
from ir_utils.utils import train_types
from ir_utils.loss_utils import avg_norm_jac
import ir_utils.dataloaders as dataloaders

# this function runs both training and test passes through data
def datapass(dataloader, train=True):
    if train:
        net.train()
    else:
        net.eval()
    
    num_correct = 0
    total_loss = 0
    for batch_idx, (samples, labels) in enumerate(dataloader):
        samples, labels = samples.to(device), labels.to(device)
        if train and args.train_type == 'at':
            samples = adversary.perturb(samples, labels)
        
        outputs = net(samples)
        if args.model_name == 'SimpleCNN':
            outputs = net.probabilities
            
        loss = F.cross_entropy(outputs, labels)
        
        if args.train_type == 'jr':
            loss += args.lambda_jr * avg_norm_jac(net, samples, args.n_classes, args.gpu)
        
        preds = torch.argmax(outputs, dim=1)
        num_correct += torch.eq(preds, labels).sum().item()
        total_loss += loss.item()
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return num_correct / len(dataloader.dataset), total_loss / len(dataloader)

# set device
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# get dataloaders
if args.dataset == 'CIFAR-10':
    train_loader, test_loader = dataloaders.cifar10()
elif args.dataset == 'MNIST':
    train_loader, test_loader = dataloaders.mnist()
    
if args.norm == 'inf':
    norm = np.inf
elif args.norm == '2':
    norm = 2

print(f'Training {args.n_seeds} {args.model_name} models with {train_types[args.train_type]}.\n')

for model_num in range(args.n_seeds):
    model_path = f'{args.save_dir}/{args.dataset}/{args.model_name}_{args.train_type}/model{model_num}'

    start = time.time()
    
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
        
    net.cuda()
    
    if args.train_type == 'at':
        adversary = PGDAttack(predict=net, loss_fn=F.cross_entropy, eps=args.epsilon,
              nb_iter=args.iters, eps_iter=args.step_size, rand_init=True,
              clip_min=args.clip_min, clip_max=args.clip_max, ord=norm, targeted=args.targeted)
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
          momentum=args.momentum, 
          weight_decay=args.wd, 
          nesterov=args.nesterov)
    
    lr_decayer = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    
    print(f'TRAINING MODEL {model_num}')
    print(f'Saving to {model_path}')

    for epoch in range(1, args.n_epochs + 1):
        train_acc, train_loss = datapass(train_loader)
        
        if epoch % args.print_freq == 0:
            test_acc, test_loss = datapass(test_loader, train=False)
            print('Epoch #{}:\tTrain loss: {:.4f}\tTrain acc: {:.4f}\tTest loss: {:.4f}\tTest acc: {:.4f}'.format(epoch, train_loss, train_acc, test_loss, test_acc))
        lr_decayer.step()
    torch.save(net.state_dict(), model_path)
    
    print(f'\nTotal train time: {(time.time()-start)/60:.1f} minutes\n\n')