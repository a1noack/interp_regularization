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
from ir_utils.loss_utils import frob_norm_jac, norm_im, cos_sim, double_backprop
import ir_utils.dataloaders as dataloaders

# this function runs both training and test passes through data
def datapass(dataloader, train=True, adv_test=False):
    if train:
        net.train()
    else:
        net.eval()
    
    n_correct = 0
    total_loss = 0
    avg_jac = 0
    for batch_idx, batch_data in enumerate(dataloader):
        if len(batch_data) == 2:
            samples, labels = [x.to(device) for x in batch_data]
        else:
            samples, labels, target_interps = [x.to(device) for x in batch_data]
            
        if (train and args.train_type == 'at') or adv_test:
            samples = adversary.perturb(samples, labels)
        
        outputs = net(samples)
            
        # F.cross_entropy combines log softmax and nll into one function
        loss = F.cross_entropy(outputs, labels)
        
        optimizer.zero_grad()
        
        # controls regularization strength
        factor = np.sin(epoch / (args.n_epochs * (2. / np.pi)))
        if args.train_type == 'jr':
            loss += args.lambda_jr * frob_norm_jac(net, samples, args.n_classes, args.gpu)
        elif args.train_type == 'ir':
            loss += (factor * args.lambda_ir) * norm_im(net, samples, labels, target_interps, args.n_classes, args.gpu)
        elif args.train_type == 'cs':
            loss += (factor * args.lambda_cs) * cos_sim(net, samples, labels, args.gpu)
        elif args.train_type == 'db':
            loss += (1. * args.lambda_db) * double_backprop(net, samples, labels, args.gpu)
        
        preds = torch.argmax(outputs, dim=1)
        n_correct += torch.eq(preds, labels).sum().item()
        total_loss += loss.item()
        
        if train:
            loss.backward()
            optimizer.step()
        
        if args.track_jac:
            avg_jac += frob_norm_jac(net, samples, args.n_classes, args.gpu, for_loss=False)
    
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
                
    return n_correct / n_samples, total_loss / n_batches, avg_jac / n_batches

# set device
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
if device != 'cpu':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# get dataloaders
if args.dataset == 'CIFAR-10':
    train_loader, test_loader = dataloaders.cifar10()
elif args.dataset == 'MNIST':
    if args.train_type == 'ir':
        train_loader, test_loader = dataloaders.mnist_interps(smoothgrad=False)
    else:
        train_loader, test_loader = dataloaders.mnist()
    
if args.norm == 'inf':
    norm = np.inf
elif args.norm == '2':
    norm = 2

print(f'Training {args.n_seeds} {args.model_name} models with {train_types[args.train_type]}.\n')

for model_num in range(args.n_seeds):
    model_path = f'{args.save_dir}/{args.dataset}/{args.model_name}_{args.train_type}/'
    if args.train_type == 'ir':
        model_path = model_path + f'model_ir{args.lambda_ir}_{model_num}'
    elif args.train_type == 'jr':
        model_path = model_path + f'model_jr{args.lambda_jr}_{model_num}'
    elif args.train_type == 'at':
        model_path = model_path + f'model_pgd{norm}_eps{args.epsilon}_iters{args.iters}_{model_num}'
    elif args.train_type == 'cs':
        model_path = model_path + f'model_cs{args.lambda_cs}_{model_num}'
    elif args.train_type == 'db':
        model_path = model_path + f'model_db{args.lambda_db}_{model_num}'
    else:
        print(f'This is not a supported training type: {args.train_type}.')

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
        train_acc, train_loss, norm_jac = datapass(train_loader)
        
        if epoch % args.print_freq == 0:
            test_acc, test_loss, norm_jac = datapass(test_loader, train=False)
            print(f'Epoch #{epoch}:\tTrain loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tTest loss: {test_loss:.4f}\tTest acc: {test_acc:.4f}\tNorm Jac. {norm_jac:.4f}')
        lr_decayer.step()
    try:
        torch.save(net.state_dict(), model_path)
    except OSError:
        print('Error encountered when saving model.')
    adv_acc, _, _ = datapass(test_loader, adv_test=True)
    print(f'Adv. test acc: {adv_acc:.4f}')
    print(f'Attack is {args.attack_type}-{args.iters}, l_{args.norm} norm, epsilon = {args.epsilon}, \
      \n\tstep size = {args.step_size}, iters = {args.iters}\n')
    
    print(f'\nTotal train time: {(time.time()-start)/60:.1f} minutes\n\n')