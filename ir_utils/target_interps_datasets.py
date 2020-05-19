from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF
import torchvision
import random
import os
import os.path
import torch
from PIL import Image

class MNISTInterps(VisionDataset):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 interp_transform=None, thresh=None, permute=False):
        super(MNISTInterps, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.interp_transform = interp_transform
        self.thresh = thresh
        self.permute = permute

        if self.train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'
        self.data, self.targets, self.target_interps = torch.load(os.path.join(root, data_file))

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

    def __init__(self, root, train=True, augment=False, thresh=None, permute=False):
        super(CIFAR10Interps, self).__init__(root, transform=None, target_transform=None)
        self.train = train  # training set or test set
        self.thresh = thresh
        self.permute = permute
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
    
    def joint_transform(self, img, target_interp):
        # convert to PIL image for transforms
        img = TF.to_pil_image(img)
        target_interp = TF.to_pil_image(target_interp)
        
        # pad before cropping
        img = TF.pad(img, 4, fill=0)
        target_interp = TF.pad(target_interp, 4, fill=0)
        
        # random crop img and target_interp identically
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            img, output_size=(32, 32))
        img = TF.crop(img, i, j, h, w)
        target_interp = TF.crop(target_interp, i, j, h, w)

        # random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            target_interp = TF.hflip(target_interp)

        # transform to tensor
        img = TF.to_tensor(img)
        target_interp = TF.to_tensor(target_interp)
        
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
        
        if self.permute:
            idx = torch.randperm(target_interp.nelement())
            target_interp = target_interp.view(-1)[idx].view(target_interp.shape)

        return img, target, target_interp

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}