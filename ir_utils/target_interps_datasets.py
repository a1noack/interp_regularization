from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision
import random
import os
import os.path
import torch
from PIL import Image

def variable_permute(target_interps, perm_percent):
    """Permutes perm_percent of the indices in target_interps."""
    n_elements = target_interps[0].nelement()
    n_perm = int(n_elements * perm_percent)
    shape = target_interps[0].shape

    for i in range(len(target_interps)):
        target_interp_i = target_interps[i].view(-1)
        idxs_to_perm = torch.randperm(n_elements)[:n_perm]
        values = target_interp_i[idxs_to_perm]

        permed_values = values[torch.randperm(n_perm)]

        target_interp_i[idxs_to_perm] = permed_values

        target_interps[i] = target_interp_i.reshape(shape)

class MNISTInterps(VisionDataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 interp_transform=None, thresh=None, permute_percent=0.):
        super(MNISTInterps, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.interp_transform = interp_transform
        self.thresh = thresh
        self.permute_percent = permute_percent

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        self.data, self.targets, self.target_interps = torch.load(os.path.join(root, data_file))
        
        if permute_percent > 0:
            variable_permute(self.target_interps, permute_percent)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, target_interp) where target is index of the target class and target 
                interp is the target saliency map for the prediction.
        """
        img, target, target_interp = self.data[index], int(self.targets[index]), self.target_interps[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.interp_transform is not None:
            target_interp = self.interp_transform(target_interp)
        
        if self.thresh != None:
            std = target_interp.std()
            mean = target_interp.mean()
            target_interp = torch.where(target_interp > mean + self.thresh * std, target_interp, torch.tensor([0.]).cpu())

        return img, target, target_interp

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}
    

class CIFAR10Interps(VisionDataset):
    classes = ['0 - airplance', '1 - automobile', '2 - bird', '3 - cat', '4 - deer', 
               '5 - dog', '6 - frog', '7 - horse', '8 - ship', '9 - truck']

    def __init__(self, root, train=True, augment=False, thresh=None, permute_percent=0.):
        super(CIFAR10Interps, self).__init__(root, transform=None, target_transform=None)
        self.train = train  # training set or test set
        self.thresh = thresh
        self.permute_percent = permute_percent
        self.augment = augment

        if self.train:
            file_names = [f'training{i}.pt' for i in range(5000, 50001, 5000)]
        else:
            file_names = ['test5000.pt', 'test10000.pt']
            
        data = []
        targets = []
        target_interps = []
        for file_name in file_names:
            data_, targets_, target_interps_ = torch.load(os.path.join(root, file_name))
            
            data.append(data_)
            targets.append(targets_)
            target_interps.append(target_interps_)

        self.data = torch.cat(data, dim=0)
        self.targets = torch.cat(targets, dim=0)
        self.target_interps = torch.cat(target_interps, dim=0)
        
        if self.train:
            train_type = 'training'
        else:
            train_type = 'test'
            
        print(f'Target interps threshold: {thresh}')
        print(f'Permute percent for {train_type} set target interps: {permute_percent*100:.2f}%')
        if permute_percent > 0:
            variable_permute(self.target_interps, permute_percent)
    
    def joint_transform(self, img, target_interp):
        img = F.pad(img, (4, 4, 4, 4), 'constant', 0)
        target_interp = F.pad(target_interp, (4, 4, 4, 4), 'constant', 0)
        
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=(32, 32))
        # this is assuming that the image and target interps are in CxHxW format
        img = img[:, i:i+h, j:j+w] 
        target_interp = target_interp[:, i:i+h, j:j+w]
        
        if random.random() > 0.5:
            # images in CxHxW format, so to do horizontal flip, flip along width dimension, i.e. dim=2
            img = torch.flip(img, dims=[2]) 
            target_interp = torch.flip(target_interp, dims=[2])
        
        return img, target_interp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, target_interp) where target is index of the target class and target 
                interp is the target saliency map for the prediction.
        """
        img, target, target_interp = self.data[index], int(self.targets[index]), self.target_interps[index]
        
        if self.augment:
            img, target_interp = self.joint_transform(img, target_interp)
        
        if self.thresh != None:
            std = target_interp.std()
            mean = target_interp.mean()
            target_interp = torch.where(target_interp > mean + self.thresh * std, target_interp, torch.tensor([0.]).cpu())

        return img, target, target_interp

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}