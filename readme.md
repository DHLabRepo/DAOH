## Direct amplitude-only hologram realized by broken symmetry

### Requirements
Required packages to reproduce the results are as follows:

- numpy==1.22.2
- torch==1.13.0
- opencv-contrib-python==4.6.0.66
- torchvision==0.14.0
- kornia==0.7.0
- tqdm==4.64.1
- imageio==2.26.0
- pandas==1.4.3

Required system to run the code includes: 
- Nvidia graphic card
- System memory > 140 GB

### Evaluation
To generate hologram and reconstruction images, run the following command:

```bash
python main.py --slice <start_y> <end_y> <start_x> <end_x>
```

Example:
```bash
python main.py --slice 350 850 550 1050
```

This `main.py` script will:
- Generate `img_out/hologram.png`, `img_out/recon_without_subpixel.png`, and `img_out/recon_with_subpixel.png` images from `img_in/img.png`.
- Print the PSNR and SSIM values of the reconstructed images compared to the target image.
- Use the `slice` option to simulate only a partial image due to the high memory usage of sub-pixel structure simulation.

The default slice values are `[350, 850, 550, 1050]`, which is set for the default input image.

### Output
The script will produce the following output files:
- `img_out/hologram.png`: The generated hologram.
- `img_out/recon_without_subpixel.png`: The reconstructed image without subpixel simulation.
- `img_out/recon_with_subpixel.png`: The reconstructed image with subpixel simulation.

### Metrics
The script will print the PSNR and SSIM values for the reconstructed images, both with and without subpixel simulation.