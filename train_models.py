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
from ir_utils.utils import train_types, get_model_path, instantiate_model
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
            cs_loss, gm_loss = cos_sim(net, samples, labels, args.gpu)
            loss += factor * ((args.lambda_cs * cs_loss) + (args.lambda_gm * gm_loss))
        elif args.train_type == 'db':
            loss += (1. * args.lambda_db) * double_backprop(net, samples, labels, args.gpu)
        
        preds = torch.argmax(outputs, dim=1)
        n_correct += torch.eq(preds, labels).sum().item()
        total_loss += loss.item()
        
        if train:
            loss.backward()
            optimizer.step()
        
        if args.track_jac:
            # When performing this calculation for CIFAR-10, 128*10 examples need to be created. This is too much.
            # So, just take the first 20 examples and use them to calculate the Jacobian
            avg_jac += frob_norm_jac(net, samples[:20], args.n_classes, args.gpu, for_loss=False)
    
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

    print(f'Training {args.model_name} model with {train_types[args.train_type]}.\n')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  
    
    # construct file path to save location and instantiate model
    model_path = get_model_path(args)
    
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
    for epoch in range(1, args.n_epochs + 1):
        train_acc, train_loss, norm_jac = datapass(train_loader)

        if epoch % args.print_freq == 0:
            test_acc, test_loss, norm_jac = datapass(test_loader, train=False)
            print(f'Epoch #{epoch}:\tTrain loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tTest loss: {test_loss:.4f}\tTest acc: {test_acc:.4f}\tNorm Jac. {norm_jac:.4f}')
        lr_decayer.step()
    print(f'\nTotal train time: {(time.time()-start)/60:.1f} minutes\n\n')

    # save final model
    try:
        torch.save(net.state_dict(), model_path)
    except OSError:
        print('Error encountered when saving model.')

    # get final test and adversarial accuracy
    adv_acc, _, _ = datapass(test_loader, adv_test=True)
    print(f'Adv. test acc: {adv_acc:.4f}')
    print(f'Attack is {args.attack_type}-{args.iters}, l_{args.norm} norm, epsilon = {args.epsilon}, \
      \n\tstep size = {args.step_size}, iters = {args.iters}\n')
