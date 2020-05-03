import torch
import numpy as np
import torch.nn.functional as F
import ir_utils.wide_resnet as wide_resnet
import ir_utils.dataloaders as dataloaders
from advertorch.attacks import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack
from ir_utils.cmd_args import cmd_args as args
from ir_utils.utils import train_types, mean_confidence_interval
from ir_utils import simple_models

def test():
    correct = 0
    adv_correct = 0
    
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
    
    adv_test_accuracies.append(100. * float(adv_correct / len(test_loader.dataset)))
    test_accuracies.append(100. * float(correct / len(test_loader.dataset)))

# set device
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# get dataloaders
if args.dataset == 'CIFAR-10':
    train_loader, test_loader = dataloaders.cifar10()
elif args.dataset == 'MNIST':
    train_loader, test_loader = dataloaders.mnist()
    
adv_test_accuracies = []
test_accuracies = []

if args.norm == 'inf':
    norm = np.inf
elif args.norm == '2':
    norm = 2

print(f'Attacking {args.n_seeds} {args.model_name} models with {train_types[args.train_type]}')
print(f'Attack is {args.attack_type}-{args.iters}\n')

for i in range(args.n_seeds):
    # create network structure
    if args.model_name == 'WRN-28-10':
        net = wide_resnet.Wide_ResNet(depth=28, widen_factor=10, 
              dropout_rate=args.dropout, 
              num_classes=args.n_classes)
    elif args.model_name == 'WRN-16-10':
        net = wide_resnet.Wide_ResNet(depth=16, widen_factor=10, 
                                      dropout_rate=args.dropout, 
                                      num_classes=args.n_classes)
    elif args.model_name == 'SimpleCNN':
        net = simple_models.SimpleCNN()
    
    # load saved model's parameters
    model_path = f'{args.save_dir}/{args.dataset}/{args.model_name}_{args.train_type}/model{i}'
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print(f'CANNOT LOAD MODEL AT: {model_path}')
        continue
    
    net.cuda()
    net.eval()
    
    # instantiate adversary
    if args.attack_type == 'PGD':
        adversary = PGDAttack(predict=net, loss_fn=F.cross_entropy, eps=args.epsilon,
             nb_iter=args.iters, eps_iter=args.step_size, rand_init=True,
             clip_min=args.clip_min, clip_max=args.clip_max, ord=norm, targeted=args.targeted)
    elif args.attack_type == 'CW':
        adversary = CarliniWagnerL2Attack(predict=net, num_classes=args.n_classes, confidence=0,
             targeted=args.targeted, learning_rate=0.01,
             binary_search_steps=9, max_iterations=args.iters,
             clip_min=args.clip_min, clip_max=args.clip_max)
    
    print(f'attacking model {i}')
    test()

# get adversarial test and standard accuracy
adv_ci = mean_confidence_interval(adv_test_accuracies)
std_ci = mean_confidence_interval(test_accuracies)

print(f'\nadversarial test accuracy: {adv_ci[0]:.2f} +/- {adv_ci[1]:.2f}')
print(f'standard test accuracy: {std_ci[0]:.2f} +/- {std_ci[1]:.2f}')