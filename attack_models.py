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
        samples, labels = samples.to(device), labels.to(device)
                
        # for perturbed samples
        adv_samples = adversary.perturb(samples, labels)
        adv_outputs = net(adv_samples)
        adv_preds = torch.argmax(adv_outputs, dim=1)
        adv_correct += torch.eq(adv_preds, labels).sum().item()
        
        # for unperturbed test samples
        outputs = net(samples)
        preds = torch.argmax(outputs, dim=1)
        correct += torch.eq(preds, labels).sum().item()
        
        # track norm of Jacobian
        sum_jac += frob_norm_jac(net, samples[:20], args.n_classes, args.gpu, for_loss=False).item()
    
    n_samples = len(test_loader.dataset)
    
    adv_test_accuracies.append(100. * float(adv_correct / n_samples))
    test_accuracies.append(100. * float(correct / n_samples))
    avg_jacs.append(sum_jac / n_samples)

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

print(f'Attacking {args.model_name} models with {utils.train_types[args.train_type]}')
print(f'Attack is {args.attack_type}-{args.iters}, l_{args.norm} norm, epsilon = {args.epsilon}, \
      \n\tstep size = {args.step_size}, iters = {args.iters}\n')

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
            if 'state_dict' in d:
                net.load_state_dict(d['state_dict'])
            else:
                net.load_state_dict(d)
        except:
            print(f'CANNOT LOAD MODEL AT: {model_path}')
            continue

        net.cuda()
        net.eval()

        # instantiate adversary
        if args.attack_type == 'PGD':
            adversary = PGDAttack(predict=net, loss_fn=F.cross_entropy, eps=args.epsilon,
                 nb_iter=args.iters, eps_iter=args.step_size, rand_init=True,
                 clip_min=args.clip_min, clip_max=args.clip_max, ord=norm, targeted=False)
        elif args.attack_type == 'CW':
            adversary = CarliniWagnerL2Attack(predict=net, num_classes=args.n_classes, confidence=0,
                 targeted=args.targeted, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=args.iters,
                 clip_min=args.clip_min, clip_max=args.clip_max)

        for j in range(args.n_attack):
            print(f'\tAttacking: {seed}')
            test()
        print('')

    # get adversarial test and standard accuracy
    adv_ci = utils.mean_confidence_interval(adv_test_accuracies)
    std_ci = utils.mean_confidence_interval(test_accuracies)
    jac_ci = utils.mean_confidence_interval(avg_jacs)

    print(f'\n\tAdversarial test accuracy: {adv_ci[0]:.2f} +/- {adv_ci[1]:.2f}')
    print(f'\tStandard test accuracy: {std_ci[0]:.2f} +/- {std_ci[1]:.2f}\n')
    print(f'\tAverage norm of Jacobian per sample: {jac_ci[0]:.6f} +/- {jac_ci[1]:.6f}\n')