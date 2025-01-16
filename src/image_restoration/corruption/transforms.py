import torch
import random
from src.image_restoration.corruption.ocean import *


class ApplyOceanCorruption:
    def __init__(self, 
                 Lx=64, 
                 Lz=64, 
                 size=(256, 256), 
                 device='cuda'):
        """
        PyTorch transform to apply ocean-related visual corruptions to an image.

        Args:
            Lx (float): Size of the ocean patch in x direction (m).
            Lz (float): Size of the ocean patch in z direction (m).
            size (tuple): Resolution of the ocean patch (M, N).
            device (str): The device to use for processing.
        """
        self.Lx = Lx
        self.Lz = Lz
        self.size = size
        self.device = device

    def _sample_parameters(self):
        """
        Randomly sample parameters for the ocean corruption transform.
        """
        wind = torch.tensor([10, 10]).float() # torch.tensor([random.uniform(0, 50), random.uniform(0, 50)]).float()
        t=0
        wind_alignment = 6 # random.uniform(0, 10)
        wave_dampening = 0.01 # random.uniform(0, 1)
        depth = 2.2 #random.uniform(2.2, 2.5)
        red = random.uniform(0, 0.6)
        water_albedo_bot = torch.tensor([random.uniform(0, 0.10), red + 0.3, red + 0.4]) # torch.tensor([random.uniform(0, 0.10), random.uniform(0.30, 0.40), random.uniform(0.60, 0.80)]) # torch.tensor([0.10, 0.38, 0.60])
        light = torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        intensity = 2.0 # random.uniform(0, 5)
        light_ambient = random.uniform(0.05, 0.1)
        light_scatter = random.uniform(0, 0.1)
        light_specular_mult = random.uniform(0.5, 1)
        light_specular_gain = random.uniform(0, 10)
        light_specular_gpow = 5.0
        wave_amplitude = 256

        # for item in [wind, t, wind_alignment, wave_dampening, depth, light, intensity, light_ambient, light_scatter, light_specular_mult, light_specular_gain, light_specular_gpow, wave_amplitude]:
        #     print(item)

        return wind, t, wind_alignment, wave_dampening, depth, water_albedo_bot, light, intensity, light_ambient, light_scatter, light_specular_mult, light_specular_gain, light_specular_gpow, wave_amplitude

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply the ocean corruption transform to an image.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Corrupted image tensor of shape (C, H, W).
        """
        # Sample parameters
        wind, t, wind_alignment, wave_dampening, depth, water_albedo_bot, light, intensity, light_ambient, light_scatter, light_specular_mult, light_specular_gain, light_specular_gpow, wave_amplitude = self._sample_parameters()

        # Generate the ocean patch
        patch = generate_ocean_patch(
            self.Lx, self.Lz, self.size[0], self.size[1], wind, t, 
            wind_alignment=wind_alignment, 
            wave_dampening=wave_dampening,
            wave_amplitude=wave_amplitude
        )

        # Normalize light and scale by intensity
        light = light / torch.norm(light) * intensity

        # Depth map for light interaction
        depth_map = torch.full(self.size, depth)

        image_corrupted = apply_corruption_ocean(
            patch,
            image,
            depth_map,
            light,
            water_albedo_bot,
            light_ambient=light_ambient,
            light_scatter=light_scatter,
            light_specular_gain=light_specular_gain,
            light_specular_gpow=light_specular_gpow,
            device=self.device
        )

        return image_corrupted