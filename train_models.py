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
from ir_utils.utils import train_types, get_path, instantiate_model
from ir_utils.loss_utils import frob_norm_jac, norm_im, cos_sim_img, double_backprop
import ir_utils.dataloaders as dataloaders
from jacobian import JacobianReg

# this function runs both training and test passes through data
def datapass(dataloader, train=True, adv_test=False):
    if train:
        net.train()
    else:
        net.eval()
    
    n_correct = 0
    total_loss = 0
    avg_jac = 0
#     start_time = time.time()
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
            # controls regularization strength
            factor = np.sin(epoch / (args.n_epochs * (2. / np.pi)))
            if args.train_type == 'jr':
#                 loss += args.lambda_jr * frob_norm_jac(net, samples, args.n_classes)
                loss += args.lambda_jr * jreg(samples, logits)
            elif args.train_type == 'ir':
                loss += (factor * args.lambda_ir) * norm_im(samples, logits, labels, target_interps)
            elif args.train_type == 'cs':
                cs_loss, gm_loss = cos_sim_img(net, samples, labels)
                loss += factor * ((args.lambda_cs * cs_loss) + (args.lambda_gm * gm_loss))
            elif args.train_type == 'db':
                loss += (1. * args.lambda_db) * double_backprop(loss, samples)
        
        preds = torch.argmax(logits, dim=1)
        n_correct += torch.eq(preds, labels).sum().item()
        total_loss += loss.item()
        
        if train:
            loss.backward()
            optimizer.step()
        
        if args.track_jac:
            # When performing this calculation for CIFAR-10, 128*10 examples need to be created. This is too much.
            # So, just take the first n_track examples and use them to calculate the Jacobian
            n_track = 20
            avg_jac += frob_norm_jac(net, samples[:n_track], args.n_classes, args.gpu, for_loss=False) / n_track
#     print(f'one epoch runtime: {time.time() - start_time}')
    
    n_samples = len(dataloader.dataset)
    n_batches = len(dataloader)
                
    return n_correct / n_samples, total_loss / n_batches, avg_jac / n_batches

if __name__ == '__main__':
    # set device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    if device != 'cpu':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # get dataloaders
    if args.dataset == 'CIFAR-10':
        if args.train_type == 'ir':
            train_loader, test_loader = dataloaders.cifar10_interps(
                batch_size=args.batch_size, thresh=1., augment=True
            )
        else:
            train_loader, test_loader = dataloaders.cifar10(batch_size=args.batch_size)
    elif args.dataset == 'MNIST':
        if args.train_type == 'ir':
            train_loader, test_loader = dataloaders.mnist_interps(batch_size=args.batch_size, smoothgrad=False)
        else:
            train_loader, test_loader = dataloaders.mnist(batch_size=args.batch_size)

    if args.norm == 'inf':
        norm = np.inf
    elif args.norm == '2':
        norm = 2
        
    if args.train_type == 'jr':
        jreg = JacobianReg()

    print(f'Training {args.model_name} model with {train_types[args.train_type]}.\n')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  
    
    # construct file path to save location and instantiate model
    model_path = get_path(args)
    
    # instantiate model, adversary, optimizer
    net = instantiate_model(args)
    net.cuda()

    adversary = PGDAttack(predict=net, loss_fn=F.cross_entropy, eps=args.epsilon,
          nb_iter=args.iters, eps_iter=args.step_size, rand_init=True,
          clip_min=args.clip_min, clip_max=args.clip_max, ord=norm, targeted=args.targeted)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
          momentum=args.momentum, 
          weight_decay=args.wd, 
          nesterov=args.nesterov)

    lr_decayer = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)

    # training loop
    print(f'TRAINING MODEL {args.seed}')
    print(f'Saving to {model_path}')
    start = time.time()
    best_acc = 0
    for epoch in range(1, args.n_epochs + 1):
        train_acc, train_loss, norm_jac = datapass(train_loader)

        if epoch % args.update_freq == 0:
            test_acc, test_loss, norm_jac = datapass(test_loader, train=False)
            print(f'Epoch #{epoch}:\tTrain loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tTest loss: {test_loss:.4f}\tTest acc: {test_acc:.4f}\tNorm Jac. {norm_jac:.4f}')
            
            if test_acc > best_acc:
                best_acc = test_acc
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
                
        lr_decayer.step()
    print(f'\nTotal train time: {(time.time()-start)/60:.1f} minutes\n\n')

    # get final test and adversarial accuracy
    adv_acc, _, _ = datapass(test_loader, adv_test=True)
    print(f'Adv. test acc: {adv_acc:.4f}')
    print(f'Attack is {args.attack_type}-{args.iters}, l_{args.norm} norm, epsilon = {args.epsilon}, \
      \n\tstep size = {args.step_size}, iters = {args.iters}\n')
