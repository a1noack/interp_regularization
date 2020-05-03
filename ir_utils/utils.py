import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

train_types = {'st':'standard training', 
               'at':'adversarial training', 
               'jr':'jacobian regularization', 
               'ir': 'interpretation regularization'}

def mean_confidence_interval(data, confidence=0.95):
    if len(data) > 0:
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, h
    return -1,-1

def threshold(a, scale):
    """Used to threshold a tensor."""
    std = a.std()
    mean = a.mean()
    a = torch.where(a > mean + scale * std, a, torch.tensor([0.]))
    return a 

def display(img, size=3):
    """Displays image with dimensions (c,h,w)"""
    img = img.squeeze()
    if img.shape[0] == 3:
        img = img.permute(1,2,0)
    fig = plt.figure(figsize=(size, size), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    ax1 = fig.add_subplot(spec[0,0])
    im1 = ax1.imshow(img.squeeze().numpy(), cmap='gist_heat')
    plt.axis('on')
    
    plt.show()