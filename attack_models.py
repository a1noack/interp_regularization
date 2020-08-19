import time
import torch
import numpy as np
import torch.nn.functional as F
import ir_utils.wide_resnet as wide_resnet
import ir_utils.dataloaders as dataloaders
from advertorch.attacks import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack
from ir_utils.cmd_args import cmd_args as args
import ir_utils.utils as utils
from ir_utils import simple_models
from ir_utils.loss_utils import frob_norm_jac

def test():
    correct = 0
    adv_correct = 0
    sum_jac = 0
    
    for samples, labels in test_loader:
        samples_clone = samples.clone()
        samples, samples_clone, labels = samples.to(device), samples_clone.to(device), labels.to(device)
                
        # for perturbed samples
        adv_samples = adversary.perturb(samples, labels)
        adv_outputs = net(adv_samples)
        adv_preds = torch.argmax(adv_outputs, dim=1)
        adv_correct += torch.eq(adv_preds, labels).sum().item()
        
        # for unperturbed test samples
        outputs = net(samples_clone)
        preds = torch.argmax(outputs, dim=1)
        correct += torch.eq(preds, labels).sum().item()
        
        # track norm of Jacobian
        sum_jac += 0 #frob_norm_jac(net, samples[:20], args.n_classes, for_loss=False).item()
    
    n_samples = len(test_loader.dataset)
    
    adv_acc = 100. * float(adv_correct / n_samples)
    std_acc = 100. * float(correct / n_samples)
    avg_jac = sum_jac / n_samples
    
    print(f'\t\tstd acc: {std_acc:.2f}, adv acc: {adv_acc:.2f}')
    
    adv_test_accuracies.append(adv_acc)
    test_accuracies.append(std_acc)
    avg_jacs.append(avg_jac)

# set device
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
if device != 'cpu':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# get dataloaders and attack parameters
if args.dataset == 'CIFAR-10':
    train_loader, test_loader = dataloaders.cifar10()
    attack_names = utils.CIFAR10_ATTACKERS
elif args.dataset == 'MNIST':
    train_loader, test_loader = dataloaders.mnist()
    attack_names = utils.MNIST_ATTACKERS

# create network structure
net = utils.instantiate_model(args)
# get path to network
dir_path = utils.get_path(args, dir_path=True)
model_dict = utils.scrape_dir(dir_path)

print('Model categories available to attack in this directory: ')
for i, model_name in enumerate(model_dict):
    print(f'{i}: {model_name}')
# get numbers of models to attack from user
to_run = [int(n) for n in input('\nEnter numbers of model categories to attack separated by spaces. Press return to attack all:\n').split()]
if len(to_run) == 0: to_run = list(range(len(model_dict)))
    
print('\nAvailable attackers: ')
for i, attack_name in enumerate(attack_names):
    print(f'{i}: {attack_name}')
# get numbers of attacks to use on each model
attack_nums = [int(n) for n in input('\nEnter numbers of attack names. Press return to use all attacks:\n').split()]
if len(attack_nums) == 0: attack_nums = list(range(len(attack_names)))
    
n_repeat = int(input('\nEnter the number of times to repeat attack for each model:\n'))

seconds_to_delay = int(input('\nEnter the number of hours to sleep before starting attacks:\n'))*60*60
print('\nsleeping...')
time.sleep(seconds_to_delay) # this is here so we can start attacks on models that haven't finished training yet

for j, attack_name in enumerate(attack_names):
    if j not in attack_nums:
        continue
    attack_configs = attack_names[attack_name]
    print('\n\n\n-----------Attacking models with: {}-----------'.format(attack_configs))
    for i, model_name in enumerate(model_dict):
        # only attack selected models
        if i not in to_run: 
            continue

        print(f'\nAttacking category {i}: {model_name}')

        adv_test_accuracies = []
        test_accuracies = []
        avg_jacs = []

        for seed in model_dict[model_name]:
            model_path = f'{dir_path}/{seed}'

            try:
                d = torch.load(model_path, map_location=device)
                net.load_state_dict(d)
                print(f'\tloaded model at {model_path}')
            except:
                print(f'CANNOT LOAD MODEL AT: {model_path}')
                continue

            net.cuda()
            net.eval()

            # instantiate adversary
            if args.attack_type == 'PGD':
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

            for h in range(n_repeat):
                print(f'\tAttacking {seed}. ({h+1}/{n_repeat})')
                test()
            net.cpu()
            print('')

        # get adversarial test and standard accuracy
        adv_ci = utils.mean_confidence_interval(adv_test_accuracies)
        std_ci = utils.mean_confidence_interval(test_accuracies)
#         jac_ci = utils.mean_confidence_interval(avg_jacs)

        print(f'\n\tAdversarial test accuracy: {adv_ci[0]:.2f} +/- {adv_ci[1]:.2f}')
        print(f'\tStandard test accuracy: {std_ci[0]:.2f} +/- {std_ci[1]:.2f}')
#         print(f'\tAverage norm of Jacobian per sample: {jac_ci[0]:.6f} +/- {jac_ci[1]:.6f}\n')