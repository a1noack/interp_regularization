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

def mnist_interps(batch_size=64, smoothgrad=True, thresh=1.):
    if smoothgrad:
        interp_type = 'smoothgrad'
    else:
        interp_type = 'simple_gradient'
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        tid.MNISTInterps('data/MNIST/SimpleCNN_at_{}'.format(interp_type), train=True, thresh=thresh), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        tid.MNISTInterps('data/MNIST/SimpleCNN_at_{}'.format(interp_type), train=False, thresh=thresh),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def cifar10_interps(batch_size=128, augment=True, thresh=1.):
    train_loader = torch.utils.data.DataLoader(
        tid.CIFAR10Interps('data/CIFAR-10/pgdL2_eps1.2549_iters7_smoothgrad/', augment=augment, thresh=thresh), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        tid.CIFAR10Interps('data/CIFAR-10/pgdL2_eps1.2549_iters7_smoothgrad/', augment=False, thresh=thresh),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader