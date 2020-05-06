import torch
import torch.distributions as tdist
import torch.nn.functional as F

def smoothgrad(net, sample, label, normalize=True, j=50, scale=1.):
    """Creates a SmoothGrad salience map for the given sample.
    
    Args:
        net: the DNN, a torch.nn.Module instance
        sample: a single sample
        label: the corresponding label for sample
        normalize: if True, normalize the magnitudes of the salience map 
        j: the number of random samples around the given sample to consider 
            when averaging simple gradient salience maps
        scale: the standard deviation of the normal distribution from which we
            select our noise to add to the original sample
    Returns: 
        A SmoothGrad salience map for sample and net.
    """
    sample = torch.autograd.Variable(sample.unsqueeze(0), requires_grad=True)
    
    # add random noise to original sample
    normal = tdist.Normal(loc=torch.tensor([0.]), scale=torch.tensor([scale]))
    shape = list(sample.shape)
    shape[0] = j
    noise = normal.sample(shape).reshape(shape)
    samples = torch.clamp(sample + noise, 0., 1.) # ensure pixels in [0,1] range
    samples = torch.cat([samples, sample], dim=0)
    
    # get salience maps
    net(samples)
    logits = net.logits
    grad_outputs = F.one_hot(label.repeat(j+1,1), num_classes=logits.shape[1]).float().squeeze()
    grads = torch.autograd.grad(logits, samples, grad_outputs=grad_outputs)[0].squeeze()
    salience_maps = torch.abs(grads)
    
    # aggregate across salience maps and within aggregated map
    salience_map = salience_maps.mean(dim=0, keepdims=True)
    if salience_map.shape[1] == 3:
        salience_map = torch.mean(salience_map, dim=1, keepdim=True)
    if normalize:
        salience_map = salience_map/torch.sum(salience_map)
        
    return salience_map.squeeze(0)

def simple_gradient(net, samples, labels, normalize=True, for_loss=False):
    """Takes a batch of samples and calculates the simple gradient salience maps for 
    each of the samples after being passed through the net.
    
    Args:
        net: the DNN, a torch.nn.Module instance
        samples: a batch of samples
        labels: the corresponding labels for the samples
        normalize: if True, normalize the magnitudes of the salience maps 
        for_loss: if True, a computation graph is built so that we can backpropagate through a gradient
    Returns: 
        A batch of simple gradient salience maps for the samples given.
    """
    samples = torch.autograd.Variable(samples, requires_grad=True)
    net(samples)
    
    grad_outputs = F.one_hot(labels, num_classes=10).float()
    grads = torch.autograd.grad(net.logits, samples, grad_outputs=grad_outputs, create_graph=for_loss)[0]
    
    salience_maps = torch.abs(grads)
    
    if salience_maps.shape[1] == 3:
        salience_maps = torch.mean(salience_maps, dim=1, keepdim=True)

    if normalize:
        salience_maps =  salience_maps / salience_maps.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        
    return salience_maps