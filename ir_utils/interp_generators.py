import torch
import torch.distributions as tdist
import torch.nn.functional as F

def smoothgrad(net, sample, label, normalize=True, j=50, scale=1.):
    """Creates smoothgrad saliency map. Unparallelized.
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
    """Parallelized version of simple gradient salience map function.
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