import numpy as np
import torch
from PIL import Image
from image_restoration.corruption.ocean import *
from image_restoration.common import ASSETS_DIR
import os

images = os.listdir("imagenet-mini")

output_dir = "imagenet_corrupted"  # Replace with output directory
os.makedirs(output_dir, exist_ok=True)

# WE MAY WANT TO VARY THESE PARAMETERS
M = 2048 # resolution
N = 2048
Lx = 64 # patch size (m)
Lz = 64
wind = torch.tensor([20, 20]).float() # wind vector (m/s)
t = 0 # time
patch = generate_ocean_patch(Lx, Lz, M, N, wind, t=t, wind_alignment=6, wave_dampening=0.05)

def load_image(name: str):
    image = Image.open(ASSETS_DIR / name)
    image = image.resize((M, N))
    image = torch.tensor(np.array(image)) / 255.0
    return image

for path in images:
    im = load_image(path)

    # WE MAY WANT TO VARY THESE
    depth = torch.full((M, N), 1) # 0.01 for caustics
    light = torch.tensor([0, -1.0, 0])
    light = light / torch.norm(light) * 2.0 # intensity
    light_ambient = 0.05
    light_scatter = 0.05
    light_specular_mult = 0.85
    light_specular_gain = 5.0
    device = 'cuda'

    image_corrupted = apply_corruption_ocean(
        patch, 
        im, 
        depth, 
        light, 
        light_ambient=light_ambient,
        light_scatter=light_scatter,
        light_specular_mult=light_specular_mult,
        light_specular_gain=light_specular_gain,
        device=device
    )
    corrupted_image_path = os.path.join(output_dir, "corrupted_"+path)
    Image.fromarray((image_corrupted.cpu().numpy() * 255).astype(np.uint8)).save(corrupted_image_path)