import torch
import torchvision

def cifar10(batch_size=128):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=True, download=True,
             transform=train_transform), 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../data', train=False, download=True,
             transform=test_transform),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader
    
def mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=True, download=True,
             transform=transform), 
        batch_size=tr_batch_size, 
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=False, download=True,
             transform=transform),
        batch_size=te_batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader