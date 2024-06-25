import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from tqdm import trange
import torchvision.transforms.functional as tvf

from wave_optics import *

def optimizer(initial, lossFunc, *args, **kwargs):
    ''' base optimizer function
    initial: initial seed tensor or list of tensors
    lossFunc: loss function. It must return loss
        ex) lossFunc(initial, *args, **kwargs)
    *args: argument for lossFunc()
    **kwargs: other arguments for optimizer and lossFunc()
        includes: optim, iterMax, lr, progress_bar    
    return: optimized tensor(complex), loss list during training
    '''
    optim = kwargs.get('optim', torch.optim.Adam)
    iterMax = kwargs.get('iterMax', 500)
    lr = kwargs.get('lr', 0.002)
    pb = kwargs.get('progress_bar', True)
    if pb:
        iters = trange(iterMax, position=0, leave=True)
    else:
        iters = range(iterMax)
        
    if isinstance(initial, list):
        zero = [init.clone() for init in initial]
        [zero0.requires_grad_(True) for zero0 in zero]
        optimizer = optim(zero, lr=lr)
    else:
        zero = initial.clone()
        zero.requires_grad_(True)
        optimizer = optim([zero], lr=lr)
    
    l = [] 
    for i in iters:
        optimizer.zero_grad()
        loss = lossFunc(zero, *args, **kwargs)
        loss.backward()
        optimizer.step()
        l.append(loss.item())
        if pb:
            iters.set_description("{}".format(loss.item()))
            iters.refresh()
            
    if isinstance(zero, list):
        return [zero0.detach() for zero0 in zero], l
    else:
        return zero.detach(), l

def quantize(tensor, min=0, max=1):
    ''' quantize 0~1 fp32 tensor to the fp32 tensor with 256 steps
    '''
    return torch.round((tensor-min)/(max-min)*255)/255*(max-min)+min

def DAOH(init, target, z, propagator, **kwargs):
    ''' synthesize DAOH
    init: initial tensor to start optimize
    target: target tensor
    z: propagation distance between the SLM plane and the target plane
    propagator: ASM propagator to use during the synthesis
    '''
    def func(init, z, target, **kwargs):
        pad_size = kwargs.get('pad_size', None)
        criterion = kwargs.get('criterion', F.mse_loss)
        prop = propagator(init.abs(), z)
        return criterion(unpad_sym_xy(prop, pad_size), target.abs())
    
    if kwargs.get('pad_size', None) is None:
        kwargs.update({'pad_size': 20})
    if kwargs.get('optim', None) is None:
        kwargs.update({'optim': torch.optim.RMSprop})
    if kwargs.get('lr', None) is None: 
        kwargs.update({'lr': 0.002})
    
    pad_value = kwargs.get('pad_value', 0.5)
    init_pad = pad_sym_xy(init.abs()*(1+1j*0), kwargs.get('pad_size', None), value=pad_value)
    r = optimizer(init_pad, func, z, target, **kwargs)
    return quantize(torch.clip(r[0].abs(), 0, 1)), r[1]