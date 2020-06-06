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
    
def mnist(batch_size=64):
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

def mnist_interps(batch_size=64, thresh=1., permute_percent=0., version=0):
    if version == 0:
        interp_type = 'smoothgrad'
    elif version == 1:
        interp_type = 'simple_gradient'
    print(f'Using target interps generated using {interp_type}')
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        tid.MNISTInterps('data/MNIST/SimpleCNN_at_{}'.format(interp_type), 
                         train=True, 
                         thresh=thresh,
                         permute_percent=permute_percent), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        tid.MNISTInterps('data/MNIST/SimpleCNN_at_{}'.format(interp_type), 
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