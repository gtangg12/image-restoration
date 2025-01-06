import torch
import random
from image_restoration.corruption.ocean import *


class ApplyOceanCorruption:
    def __init__(self, 
                 Lx=64, 
                 Lz=64, 
                 size=(2048, 2048), 
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
        wind = torch.tensor([random.uniform(0, 50), random.uniform(0, 50)]).float()
        t=0
        wind_alignment = random.uniform(0, 10)
        wave_dampening = random.uniform(0, 1)
        depth = random.uniform(0, 2.5)
        light = torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        intensity = random.uniform(0, 5)
        light_ambient = random.uniform(0.05, 1)
        light_scatter = random.uniform(0, 1)
        light_specular_mult = random.uniform(0.5, 1)
        light_specular_gain = random.uniform(0, 10)

        return wind, t, wind_alignment, wave_dampening, depth, light, intensity, light_ambient, light_scatter, light_specular_mult, light_specular_gain

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply the ocean corruption transform to an image.

        Args:
            image (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Corrupted image tensor of shape (C, H, W).
        """
        # Sample parameters
        wind, t, wind_alignment, wave_dampening, depth, light, intensity, light_ambient, light_scatter, light_specular_mult, light_specular_gain = self._sample_parameters()

        # Generate the ocean patch
        patch = generate_ocean_patch(
            self.Lx, self.Lz, self.size[0], self.size[1], wind, t, 
            wind_alignment=wind_alignment, 
            wave_dampening=wave_dampening
        )

        # Normalize light and scale by intensity
        light = light / torch.norm(light) * intensity

        # Depth map for light interaction
        depth_map = torch.full(self.size, depth)

        # Ensure image is (H, W, C) format
        if image.ndim == 3 and image.shape[0] == 3:  # Input in (C, H, W)
            image = image.permute(1, 2, 0)  # Convert to (H, W, C)

        # Apply the ocean corruption function
        corrupted_image = apply_corruption_ocean(
            patch, 
            image, 
            depth_map, 
            light, 
            light_ambient=light_ambient,
            light_scatter=light_scatter,
            light_specular_mult=light_specular_mult,
            light_specular_gain=light_specular_gain,
            device=self.device
        )

        return corrupted_image.permute(2, 0, 1)  # Convert back to (C, H, W)
