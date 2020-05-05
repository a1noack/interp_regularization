import torch
from torch.autograd import Variable
from ir_utils.interp_generators import simple_gradient

def avg_norm_jac(net, samples, n_classes, device, for_loss=True):
    batch_size = samples.shape[0]
    samples = Variable(samples.repeat_interleave(n_classes, 0), requires_grad=True).to(device)
    net(samples)
    
    bp_mat = torch.eye(n_classes).repeat(batch_size, 1).to(device)    
    samples_grad = torch.autograd.grad(net.logits, samples, grad_outputs=bp_mat, create_graph=for_loss)[0]
    
    return samples_grad.pow(2).sum().sqrt()

def avg_norm_im(net, samples, labels, target_interps, n_classes, device):
    interps = simple_gradient(net, samples, labels, normalize=False, for_loss=True).squeeze()
    interp_match_loss = torch.norm(torch.abs(interps - target_interps))
    
    return interp_match_loss