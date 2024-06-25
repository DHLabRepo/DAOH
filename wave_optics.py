import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from collections.abc import Iterable
from functools import wraps, lru_cache
import cv2
from copy import deepcopy

def meta_to_tuple(meta):
    for key in meta:
        if isinstance(meta[key], Iterable):
            meta[key] = tuple(meta[key])
    return meta

def fftshift(tensor, dims=(-2,-1)):
    shifts = [tensor.size(dim)//2 for dim in dims]
    return torch.roll(tensor, shifts, dims)

def ifftshift(tensor, dims=(-2,-1)):
    shifts = [-(tensor.size(dim)//2) for dim in dims]
    return torch.roll(tensor, shifts, dims)

def fft(tensor):
    ''' Fourier transform of object
    zero frequency is at center
    '''    
    return fftshift(torch.fft.fftn(tensor, dim=(-2,-1)))

def ifft(tensor):
    ''' inverse Fourier transform of object
    zero position is at center
    '''    
    return torch.fft.ifftn(ifftshift(tensor), dim=(-2,-1))

@lru_cache(maxsize=1)
def get_x(dx, shape, device):
    ''' Get coordinates of real domain. e.g. (x, y)
    dx: pixel pitch
    shape: (x, y) shape of coordinates
    device: torch.device e.g. cpu' or 'cuda'
    '''    
    xx = [torch.linspace(-dx*s/2, dx*s/2-dx, s, dtype=torch.float32, device=device) 
        for dx, s in zip(dx, shape[-2:])]
    return torch.meshgrid(*xx)

@lru_cache(maxsize=1)
def get_f(dx, shape, device):
    ''' Get coordinates of fourier domain. e.g. (v_x, v_y)
    dx: pixel pitch
    shape: (x, y) shape of coordinates
    device: torch.device e.g. cpu' or 'cuda'
    '''    
    ff = [torch.linspace(-1/dx/2, 1/dx/2-1/dx/s, s, dtype=torch.float32, device=device)
        for dx, s in zip(dx, shape[-2:])]
    return torch.meshgrid(*ff)

@lru_cache(maxsize=1)
def get_ASM_kernel(shape, z, dx, wl, device, band_limit=True, **kwargs):   
    f = get_f(dx, shape, device)
    if isinstance(wl, Iterable):
        gammaSqrt = torch.stack([torch.sqrt(torch.clamp((1/w)**2 - f[0]**2 - f[1]**2, min=1e-9)) for w in wl], dim=-3)
        f_max = [[1/np.sqrt((2*z*(1/dx/s))**2+1)/w for dx, s in zip(dx, shape[-2:])] for w in wl]
        
    else: 
        gammaSqrt = torch.sqrt(torch.clamp((1/wl)**2 - f[0]**2 - f[1]**2, min=1e-9))
        f_max = [1/np.sqrt((2*z*(1/dx/s))**2+1)/wl for dx, s in zip(dx, shape[-2:])]

    if band_limit:
        if isinstance(wl, Iterable):
            band_mask = torch.stack([((f[0].abs() < fM[0]) & (f[1].abs() < fM[1])).to(dtype=torch.float32) for fM in f_max], dim=-3)
        else:
            band_mask = ((f[0].abs() < f_max[0]) & (f[1].abs() < f_max[1])).to(dtype=torch.float32)
        return torch.exp(2j*np.pi*gammaSqrt*z)*band_mask
    else:
        return torch.exp(2j*np.pi*gammaSqrt*z)

def prop_ASM(tensor, z, meta, filter=None, **kwargs):
    ''' propagation by angular spectrum method
    tensor: mono color or RGB
    z: propagation distance in m.
    filter: Fourier domain filter
    '''
    meta = meta_to_tuple(meta)
    AS = fft(tensor)
    if filter is not None:
        if isinstance(filter, (list, tuple)):
            AS = torch.stack([AS.select(-3, i)*f for i,f in enumerate(filter)], -3)
        else:
            AS = AS*filter
    return ifft(AS * get_ASM_kernel(tensor.shape, z, dx=meta['dx'], wl=meta['wl'], device=tensor.device, **kwargs))

def getMaskByRange(tensor, center, width, maskType='square', **kwargs):
    ''' Build mask consisted with 1 or 0 from the range. 
    tensor: tensor to apply mask
    center: center of the mask, between -0.5 ~ 0.5, along y and x. ex) (0,0)
    width: width of the mask, between 0 ~ 1.0, along y and x. ex) (0.2,0.2)
    maskType: 'square' or 'circle'
    return: torch.tensor
    '''
    shape = tensor.shape[-2:]
 
    if(maskType=='square'):
        x = [torch.linspace(-0.5, 0.5, s, device=tensor.device) for s in shape]
        one = torch.ones(1, device=tensor.device)
        xx = [torch.heaviside(x0-(c-w/2), one)*torch.heaviside((c+w/2)-x0, one).unsqueeze(0) for x0,c,w in zip(x, center, width)]        
        mask = xx[0].T*xx[1]
        return mask
    
    if(maskType=='circle'):
        image = np.zeros(shape)
        c = tuple(np.multiply(np.add(center, 0.5), shape).astype(int))[::-1]
        w = tuple((np.multiply(width, shape)/2).astype(int))[::-1]
        image = cv2.ellipse(image, c, w, 0, 0, 360, (255,255,255), -1)
        mask = torch.Tensor(image).to(tensor.device)
        return mask/255

def maskByRange(tensor, center, width, maskType='square', **kwargs):
    ''' Build mask and apply it which is consisted with 1 or 0 from the range. 
    tensor: tensor to apply mask
    center: center of the mask, between -0.5 ~ 0.5, along y and x. ex) (0,0)
    width: width of the mask, between 0 ~ 1.0, along y and x. ex) (0.2,0.2)
    maskType: 'square' or 'circle'
    return: torch.tensor
    '''
    return getMaskByRange(tensor, center, width, maskType=maskType, **kwargs) * tensor

def batchwiseFunc(tensor, func, dim=0):
    '''Batchwise function
    eg. dim=0: batchwise min
        dim=1: batch/channelwise min
    '''    
    tmp = tensor.reshape(*tensor.shape[:dim+1], -1)
    tmp = func(tmp, dim=dim+1, keepdim=True)
    if not isinstance(tmp, torch.Tensor):
        tmp = tmp[0]    
    for _ in range(len(tensor.shape)-len(tmp.shape)):
        tmp = tmp.unsqueeze(-1)
    return tmp

def batchwiseNorm(tensor, dim=0, epsilon=1e-7):
    '''Batchwise normalization.
    eg. dim=0: batchwise normalization
        dim=1: batch/channelwise normalization
    '''    
    tmp = tensor.reshape(*tensor.shape[:dim+1], -1)
    tmp -= tmp.min(dim+1, keepdim=True)[0]
    tmp /= tmp.max(dim+1, keepdim=True)[0]+epsilon
    return tmp.reshape(*tensor.shape)

def downSample(tensor, kernel_size, stride=None):
    ''' down-sampling, size of the input tensor is reduced.
    Only integer kernel size is allowed. 
    '''
    if np.all(np.array(kernel_size)==1): # if there's no downsize, then skip.
        return tensor
    
    if stride is None:
        stride = kernel_size
    
    if tensor.dtype==torch.complex64:
        r = F.avg_pool2d(tensor.real, kernel_size, stride=stride) + 1j*F.avg_pool2d(tensor.imag, kernel_size, stride=stride)
    else:
        r = F.avg_pool2d(tensor, kernel_size, stride=stride)
    return r

def pad_sym_xy(timg, pad_size, value=0.5):
    ''' symmetrically pad tensor 
    '''    
    if isinstance(pad_size, Iterable):
        p_size1, p_size2 = pad_size[0], pad_size[1]
    else:
        p_size1, p_size2 = pad_size, pad_size
        
    if p_size1>0 or p_size2>0:
        return F.pad(timg, [p_size2,p_size2,p_size1,p_size1], value=value)
    else:
        return timg

def unpad_sym_xy(timg, pad_size):
    ''' symmetrically unpad tensor 
    '''        
    if isinstance(pad_size, Iterable):
        p_size1, p_size2 = pad_size[0], pad_size[1]
    else:
        p_size1, p_size2 = pad_size, pad_size
        
    if p_size1>0 and p_size2>0:
        return timg[..., p_size1:-p_size1, p_size2:-p_size2]
    elif p_size1>0:
        return timg[..., p_size1:-p_size1, :]
    elif p_size2>0:
        return timg[..., p_size2:-p_size2]
    else:
        return timg

def simulate_intensity(tensor, z, meta, filterPos=(0.0,0.0), filterWidth=(0.9,0.9), superSample=1, centerShift=True, padding=100, **kwargs):
    ''' simulate layer-based hologram
    tensor: Tensor
    z: propagating distance in m
    filterPos: center position of filter(-0.5~0.5). ex) (0.0,0.0)
    filterWidth: filter size in unit(0~1.0). ex) (0.35,0.35)
    superKron: specified tensor for kron the input Tensor. ex) torch.ones(2,2)
    superSample: super-sampling number. ex) 2
    centerShift: wheather compensating off-axis. ex) True or False
    maskType: mask type of spatial filter. ex) 'square' or 'circle'
    return: complex Tensor after filtering and resizing
    '''
    def _centerShift(tensor, z, meta, filterPos):
        return torch.stack([torch.roll(tensor.select(-3, i), [-int(wl/dx0*p*z/dx0+0.5) for p,dx0 in zip(filterPos, meta['dx'])], dims=(-2,-1)) 
                          for i, wl in enumerate(meta['wl'])], dim=-3)

    superKron = torch.ones(superSample, superSample, device=tensor.device)
    if superSample>1:
        superKron[...,0] = 0
        superKron[...,0,:] = 0

    pos = np.divide(filterPos, superSample)
    width = np.divide(filterWidth, superSample)
    
    tmp = pad_sym_xy(tensor, padding, value=0)

    # we simulate pixel pitch here
    new_meta = deepcopy(meta)
    new_meta['dx'] = tuple(np.divide(new_meta['dx'], superSample))
    tmp = torch.kron(tmp, superKron) if (superSample>1) else tmp
    filters = [getMaskByRange(tmp, pos, w, **kwargs) for w in [np.max(new_meta['wl'])/wl*width for wl in new_meta['wl']]]
    tmp = prop_ASM(tmp, z, meta=new_meta, filter=filters)
    tmp = downSample(tmp, (superSample, superSample)).abs()**2/superKron.mean()**2 if (superSample>1) else tmp.abs()**2
    tmp = unpad_sym_xy(tmp, padding)

    if centerShift:
        tmp = _centerShift(tmp, z, meta, filterPos)
    return tmp