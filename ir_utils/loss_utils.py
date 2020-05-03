import torch
from torch.autograd import Variable

def avg_norm_jac(net, samples, n_classes, device, for_loss=True):
    batch_size = samples.shape[0]
    samples = Variable(samples.repeat_interleave(n_classes, 0), requires_grad=True).to(device)
    net(samples)
    outputs = net.logits
    
    bp_mat = torch.eye(n_classes).repeat(batch_size, 1).to(device)    
    samples_grad = torch.autograd.grad(outputs, samples, grad_outputs=bp_mat, create_graph=for_loss)[0]
    
    return samples_grad.pow(2).sum().sqrt()