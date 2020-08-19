import torch
import torchvision
import ir_utils.target_interps_datasets as tid

def cifar10(batch_size=128, augment=True):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.ToTensor(),
    ])
    if not augment:
        train_transform = test_transform

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('data/CIFAR-10', train=True, download=True,
             transform=train_transform), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('data/CIFAR-10', train=False, download=True,
             transform=test_transform),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader
    
def mnist(batch_size=50):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/MNIST', train=True, download=True,
             transform=transform), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/MNIST', train=False, download=True,
             transform=transform),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def mnist_interps(batch_size=50, thresh=1., permute_percent=0., version=0):
    if version == 1:
        data_path = 'data/MNIST/random_net_gradients'
    elif version == 2:
        data_path = 'data/MNIST/pgdL2_eps1.5_iters40_simp_unproc'
    elif version == 3:
        data_path = 'data/MNIST/pgdL2_eps2.5_iters40_simp_unproc'
    elif version == 4:
        data_path = 'data/MNIST/pgdLinf_eps.3_iters40_simp_unproc'
    elif version == 5:
        data_path = 'data/MNIST/std_train_simp_unproc'
    elif version == 6:
        data_path = 'data/MNIST/pgdL2_eps1.5_iters40_smooth_unproc'
    elif version == 7:
        data_path = 'data/MNIST/std_train_smooth_unproc'
    else:
        raise RuntimeError(f'Target interps version {version} is not implemented yet: ')
    print('Getting target interps dataset from:', data_path)
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        tid.MNISTInterps(data_path, 
                         train=True, 
                         thresh=thresh,
                         permute_percent=permute_percent), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        tid.MNISTInterps(data_path, 
                         train=False, 
                         thresh=thresh, 
                         permute_percent=permute_percent),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def cifar10_interps(batch_size=128, augment=True, thresh=1., permute_percent=0., version=0):
    if version == 0:
        data_path = 'data/CIFAR-10/pgdL2_eps1.2549_iters7_smoothgrad/'
    elif version == 1:
        data_path = 'data/CIFAR-10/pgdL2_eps0.314_iters7_simp_unproc'
    elif version == 2:
        data_path = 'data/CIFAR-10/pgdL2_eps0.314_iters7_smooth_unproc'
    elif version == 3:
        data_path = 'data/CIFAR-10/std_train_simp_unproc'
    elif version == 4:
        data_path = 'data/CIFAR-10/std_train_smooth_unproc'
    else:
        raise RuntimeError(f'Target interps version {version} is not implemented yet: ')
    print('Getting target interps dataset from:', data_path)
        
    train_loader = torch.utils.data.DataLoader(
        tid.CIFAR10Interps(data_path, train=True, 
                           augment=augment, thresh=thresh, 
                           permute_percent=permute_percent), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        tid.CIFAR10Interps(data_path, train=False, 
                           augment=False, thresh=thresh, 
                           permute_percent=permute_percent),
        batch_size=batch_size, 
        shuffle=False,
        
    )
    
    return train_loader, test_loader