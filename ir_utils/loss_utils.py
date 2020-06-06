import torch
import torch.nn.functional as F
from torch.autograd import Variable
from ir_utils.interp_generators import simple_gradient

def frob_norm_jac(net, samples, n_classes, for_loss=True):
    """Jacubovitz's 'Jacobian Regularizer'
    
    Args:
        net: the DNN, a torch.nn.Module instance
        samples: a minibatch of samples
        n_classes: the number of classes for the dataset
        for_loss: if True, a computation graph is built so that we can backpropagate through a gradient
    Returns: 
        jac_reg_loss: the Frobenius norm of the Jacobian of the logits of the 
            net w.r.t. the inputs
    """
    batch_size = samples.shape[0]
    samples = Variable(samples.repeat_interleave(n_classes, 0), requires_grad=True)
    logits = net(samples)
    
    grad_outputs = torch.eye(n_classes).repeat(batch_size, 1)
    samples_grad = torch.autograd.grad(logits, samples, grad_outputs=grad_outputs, create_graph=for_loss)[0]
    jac_reg_loss = torch.norm(samples_grad, p='fro')
    
    return jac_reg_loss

def norm_im(samples, logits, labels, target_interps, for_loss=True):
    """Calculates the Frobenius norm of the difference between 
    target interpretations and the current simple gradient interpretations.
    
    Args:
        samples: a minibatch of samples with requires_grad = True
        logits: the logits output for samples
        labels: the labels for the samples
        target_interps: the target interpretations we are measuring distance from
        for_loss: if True, a computation graph is built so that we can backpropagate through a gradient
    Returns: 
        interp_match_loss: the Frobenius norm of the difference between 
            target interpretations and the current simple gradient interpretations
    """
    grad_outputs = F.one_hot(labels, num_classes=10).float()
    grads = torch.autograd.grad(logits, samples, grad_outputs=grad_outputs, create_graph=for_loss)[0]
    interps = torch.abs(grads)
    interp_match_loss = torch.norm(torch.abs(interps.squeeze() - target_interps.squeeze()), p='fro')
    
    return interp_match_loss

def cos_sim(samples, logits, labels, target_interps, for_loss=True, abs=False):
    """Computes losses based on (1) the size of the angle between the simple gradient interpretations
    and their corresponding samples (2) the magnitude of the simple gradient interpretations.
    
    Args:
        samples: a minibatch of samples with requires_grad = True
        logits: the logits of the network when samples passed through network
        labels: the labels for the samples
        target_interps: interpretations to align the gradients of the network at each sample with.
            Should have the same shape as samples. If not supplied, samples will be target_interps.
        for_loss: if True, a computation graph is built so that we can backpropagate through a gradient
    Returns: 
        cos_sim_loss: the average of the cosine similarity measures calculated between 
            the simple gradient salience maps and the samples
    """
    grad_outputs = F.one_hot(labels, num_classes=10).float()
    grads = torch.autograd.grad(logits, samples, grad_outputs=grad_outputs, create_graph=for_loss)[0]
    if abs:
        interps = torch.abs(grads)
    else:
        interps = grads
    cos_sim = F.cosine_similarity(target_interps.flatten(start_dim=1, end_dim=-1), 
                                  interps.flatten(start_dim=1, end_dim=-1))
    cos_sim_loss = 1 - torch.mean(cos_sim)
    # Jacubovitz et al. find norm of entire minibatch.
    grad_mag_loss = torch.norm(interps, p='fro') 
        
    return cos_sim_loss, grad_mag_loss

def double_backprop(loss, samples, for_loss=True):
    """Drucker and LeCun's 'Double Backpropagation', 
    Ross and Doshi-Velez's 'Input Gradient Regularization'
    
    Args:
        net: the DNN, a torch.nn.Module instance
        samples: a minibatch of samples  with requires_grad = True
        labels: the labels for the samples
        loss: the loss function used to calculate the gradient w.r.t. the samples
        for_loss: if True, a computation graph is built so that we can backpropagate through a gradient
    Returns: 
        double_bp_loss: the l2 norm of the gradient of the loss function evaluated at the samples
            w.r.t. the samples
    """
    samples_grad = torch.autograd.grad(loss, samples, create_graph=for_loss)[0]
    double_bp_loss = torch.norm(samples_grad, p='fro')
    
    return double_bp_loss