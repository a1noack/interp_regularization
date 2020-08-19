import time
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import PGDAttack
from jacobian import JacobianReg
import ir_utils.resnet as resnet
import ir_utils.simple_models as simple_models
import ir_utils.wide_resnet as wide_resnet
from ir_utils.cmd_args import cmd_args as args
from ir_utils.utils import train_types, get_path, instantiate_model
from ir_utils.loss_utils import frob_norm_jac, norm_im, cos_sim, double_backprop
import ir_utils.utils as utils
import ir_utils.dataloaders as dataloaders


# this function runs both training and test passes through data
def datapass(dataloader, train=True, adv_test=False):
    if train:
        net.train()
    else:
        net.eval()
    
    stats = {'n_correct':0,
             'total_loss':0,
             'total_cs_loss':0,
             'total_gm_loss':0}

    for batch_idx, batch_data in enumerate(dataloader):
        if len(batch_data) == 2:
            samples, labels = [x.to(device) for x in batch_data]
        else:
            samples, labels, target_interps = [x.to(device) for x in batch_data]
            
        if (train and args.train_type == 'at') or adv_test:
            samples = adversary.perturb(samples, labels)
            
        if args.train_type in ['jr', 'ir', 'db', 'cs']:
            samples.requires_grad = True
        
        logits = net(samples)
            
        # F.cross_entropy combines log softmax and nll into one function
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        
        if train:
            if args.train_type == 'jr':
                if args.jr_approx == 1:
                    jac_loss = jreg(samples, logits) # approximation of full Jacobian, with num. projections = 1
                else:
                    jac_loss = frob_norm_jac(net, samples, args.n_classes)
                loss += args.lambda_jr * jac_loss
            elif args.train_type == 'ir':
                im_loss = norm_im(samples, logits, labels, target_interps)
                loss += args.lambda_ir * im_loss
            elif args.train_type == 'cs':
                cs_loss, gm_loss = cos_sim(samples, logits, labels, 
                                           target_interps=target_interps, abs=args.grad_abs)
                loss += (args.lambda_cs * cs_loss) + (args.lambda_gm * gm_loss)
                stats['total_cs_loss'] += cs_loss.item()
                stats['total_gm_loss'] += gm_loss.item()
            elif args.train_type == 'db':
                db_loss = double_backprop(loss, samples)
                loss += args.lambda_db * db_loss
        
        preds = torch.argmax(logits, dim=1)
        stats['n_correct'] += torch.eq(preds, labels).sum().item()
        stats['total_loss'] += loss.item()
        
        if train:
            loss.backward()
            optimizer.step()
    
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
    
    stats['accuracy'] = stats['n_correct'] / n_samples
    stats['total_loss'] = stats['total_loss'] / n_batches
    stats['total_cs_loss'] = stats['total_cs_loss'] / n_batches
    stats['total_gm_loss'] = stats['total_gm_loss'] / n_batches
        
    return stats

if __name__ == '__main__':
    # set device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    if device != 'cpu':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f'Device = {device}')

    # get dataloaders
    if args.dataset == 'CIFAR-10':
        if args.train_type in ['ir', 'cs']:
            train_loader, test_loader = dataloaders.cifar10_interps(
                batch_size=args.batch_size, 
                thresh=args.thresh, 
                augment=True, 
                permute_percent=args.permute_percent,
                version=args.interps_version
            )
        else:
            train_loader, test_loader = dataloaders.cifar10(batch_size=args.batch_size)
        attack_configs = utils.CIFAR10_ATTACKERS[args.attack_name+'_train']
    elif args.dataset == 'MNIST':
        if args.train_type in ['ir', 'cs']:
            train_loader, test_loader = dataloaders.mnist_interps(
                batch_size=args.batch_size, 
                thresh=args.thresh,
                permute_percent=args.permute_percent, 
                version=args.interps_version
            )
        else:
            train_loader, test_loader = dataloaders.mnist(batch_size=args.batch_size)
        attack_configs = utils.MNIST_ATTACKERS[args.attack_name]

    if args.norm == 'inf':
        norm = np.inf
    elif args.norm == '2':
        norm = 2
        
    if args.train_type == 'jr' and args.jr_approx:
        jreg = JacobianReg()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # construct file path to save location and instantiate model
    model_path = get_path(args, attack_configs)
    
    # instantiate model, adversary, optimizer
    net = instantiate_model(args)
    net.cuda()

    print(f'\nTraining {args.model_name} model with {train_types[args.train_type]}.')
    
    # instantiate adversary
    if args.train_type == 'at':
        print("\tTraining against:", attack_configs)
        adversary = PGDAttack(
                predict=net, 
                loss_fn=F.cross_entropy, 
                eps=attack_configs['eps'], 
                nb_iter=attack_configs['nb_iter'], 
                eps_iter=attack_configs['eps_iter'], 
                rand_init=True, 
                clip_min=0., 
                clip_max=1., 
                ord=attack_configs['ord'], 
                targeted=False)
    elif args.train_type == 'jr':
        print(f'\tTraining with {"approximated" if bool(args.jr_approx) else "full"} Jacobian norm.')
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
          momentum=args.momentum, 
          weight_decay=args.wd, 
          nesterov=args.nesterov)

    lr_decayer = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)

    if args.train_type == 'cs':
        print(f'\targs.grad_abs, relevant in loss_utils.cos_sim, is {args.grad_abs}')
    print(f'Saving model checkpoint files to {model_path}')
    
    model_basename = os.path.basename(model_path)
    writer_path = f'runs/{args.dataset}/{model_basename}'
    print(f'Saving TensorBoard run information to {writer_path}')
    writer = SummaryWriter(log_dir=writer_path)
    
    start = time.time()
    best_acc = 0
    best_stats = {}
    for epoch in range(1, args.n_epochs + 1):
        train_stats = datapass(train_loader)

        if epoch % args.update_freq == 0:
            test_stats = datapass(test_loader, train=False)
            
            train_loss = train_stats['total_loss']
            train_acc = train_stats['accuracy']
            test_loss = test_stats['total_loss']
            test_acc = test_stats['accuracy']
            
            print(f'Epoch #{epoch}:\tTrain loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tTest loss: {test_loss:.4f}\tTest acc: {test_acc:.4f}')
            
            writer.add_scalars('Losses', {'train':train_loss, 'test':test_loss}, epoch)
            writer.add_scalars('Accuracies', {'train':train_acc, 'test':test_acc}, epoch)
            if args.train_type == 'cs':
                cs_loss = train_stats['total_cs_loss']
                gm_loss = train_stats['total_gm_loss']
                writer.add_scalar('Cossim_loss', float(cs_loss), epoch)
                writer.add_scalar('Gm_loss', float(gm_loss), epoch)
            
            if test_stats['accuracy'] > best_acc and epoch > .5 * args.n_epochs:
                best_acc = test_stats['accuracy']
                best_stats = test_stats.copy()
                best_stats.update(train_stats)
                best_stats['epoch'] = epoch
                print('\tSaving model state dict.')
                tries = 0
                while tries < 5:
                    try:
                        torch.save(net.state_dict(), model_path)
                        print('\tSave successful.')
                        tries = 5
                    except OSError:
                        tries += 1
                        print(f'\tError encountered when saving model. Try {tries}/5')
                        
        if epoch == args.n_epochs:
            final_stats = test_stats.copy()
            final_stats.update(train_stats)
            final_stats['epoch'] = epoch
                
        lr_decayer.step()
    print(f'\nTotal train time: {(time.time()-start)/60:.1f} minutes\n\n')

    # get final test and adversarial accuracy
    if args.dataset == 'CIFAR-10':
        attack_configs = utils.CIFAR10_ATTACKERS[args.attack_name]
    adversary = PGDAttack(
            predict=net, 
            loss_fn=F.cross_entropy, 
            eps=attack_configs['eps'], 
            nb_iter=attack_configs['nb_iter'], # all of the attacks we evaluate against have iters = 40, so we can hardcode this value
            eps_iter=attack_configs['eps_iter'], 
            rand_init=True, 
            clip_min=0., 
            clip_max=1., 
            ord=attack_configs['ord'], 
            targeted=False)
    net.load_state_dict(torch.load(model_path, map_location=device)) # load last saved, i.e. best, model
    std_stats = datapass(test_loader, adv_test=False, train=False)
    adv_stats = datapass(test_loader, adv_test=True, train=False)
    print('Standard test set results:', std_stats)
    print('\n\nAttacking test set with', attack_configs)
    print('Adv. test set results:', adv_stats)
    print('\n\nBest epoch stats:', best_stats)
    print('\n\nFinal epoch stats:', final_stats)
    
    out_stats = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
        args.interps_version, args.permute_percent, args.lambda_cs, 
        args.lambda_gm, best_stats['epoch'], std_stats['accuracy'], 
        adv_stats['accuracy'], best_stats['total_cs_loss'], 
        best_stats['total_gm_loss'], final_stats['total_cs_loss'],
        final_stats['total_gm_loss'])
    
    print('\n\n',out_stats,'\n\n\n')
    with open(f'{args.dataset}_{args.interps_version}_pp{args.permute_percent > 0}.csv', 'a') as f:
        f.write(out_stats+'\n')