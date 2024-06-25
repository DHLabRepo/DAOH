import argparse
import os
import torch
import numpy as np
import torchvision.transforms.functional as tvf
import matplotlib.pyplot as plt
import kornia
import cv2
import imageio
from wave_optics import *
from holograms import *
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def apply_clip_and_noise(field):
    ''' apply clipping and add noise to the input field
    '''
    field_clip = torch.clip(field.abs(), 0, 1)
    return field_clip * torch.exp(1j * field_clip**2)

def correct_intensity(timg, target):
    ''' correct the intensity of an image tensor to match the target
    '''
    def func(factor, timg, target, **kwargs):
        return torch.nn.functional.mse_loss(timg * factor, target)
    factor = torch.ones(1, device=timg.device) * target.abs().mean() / timg.abs().mean()
    r = optimizer(factor, func, timg.abs(), target.abs(), iterMax=100, progress_bar=False)
    return r[0] * timg

def cvt_img(tensor):
    ''' convert a tensor to an image format suitable for saving
    '''
    return (np.clip(np.transpose(tensor.squeeze(0).detach().cpu().numpy(), [1,2,0]), 0, 1).copy()*255).astype(np.uint8)

def generate_hologram_and_reconstruction(slice_values):
    meta = meta_to_tuple({'dx': [7.2e-6, 7.2e-6], 'wl': [638e-9, 515e-9, 460e-9]})
    superSample = 35
    z = 2e-3
    pad_size = 20
    device = 'cuda'

    timg = tvf.to_tensor(cv2.resize(imageio.imread('img_in/img.png'), (1920, 1080))[slice_values[0]:slice_values[1], slice_values[2]:slice_values[3]]).to(device).unsqueeze(0)

    propagator = lambda field, z: simulate_intensity(apply_clip_and_noise(field), z=z, meta=meta, filterWidth=[1,1], superSample=1)

    hologram = DAOH(torch.sqrt(timg) * 0.9 + 0.05, timg, z, propagator=propagator, pad_size=pad_size)[0].to('cpu')
    timg = timg.to('cpu')
    with torch.no_grad():
        sim1 = unpad_sym_xy(simulate_intensity(apply_clip_and_noise(hologram), z=z, meta=meta, filterWidth=[1, 1], superSample=1), pad_size)        
        sim2 = unpad_sym_xy(simulate_intensity(apply_clip_and_noise(hologram), z=z, meta=meta, filterWidth=[superSample, superSample], superSample=superSample), pad_size)
    sim1 = correct_intensity(sim1, timg)
    sim2 = correct_intensity(sim2, timg)

    # Save images
    os.makedirs('img_out', exist_ok=True)
    imageio.imwrite('img_out/recon_without_subpixel.png', cvt_img(sim1))
    imageio.imwrite('img_out/recon_with_subpixel.png', cvt_img(sim2))
    imageio.imwrite('img_out/hologram.png', cvt_img(hologram))
    
    # Print PSNR and SSIM
    psnr = kornia.metrics.psnr(sim1, timg, timg.abs().max()).item()
    ssim = kornia.metrics.ssim(sim1, timg, 11).mean().item()
    print(f"w/o subpixel => PSNR: {psnr} / SSIM: {ssim}")
    psnr = kornia.metrics.psnr(sim2, timg, timg.abs().max()).item()
    ssim = kornia.metrics.ssim(sim2, timg, 11).mean().item()    
    print(f"w/  subpixel => PSNR: {psnr} / SSIM: {ssim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hologram and reconstruction images.')
    parser.add_argument('--slice', type=int, nargs=4, default=[350, 850, 550, 1050], help='Slice values for input image (start_y, end_y, start_x, end_x).')
    args = parser.parse_args()

    generate_hologram_and_reconstruction(args.slice)
