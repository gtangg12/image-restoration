from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from jaxtyping import Float32
from nerfstudio.cameras.cameras import Cameras, RayBundle

from image_restoration.corruption.optics import *


np.random.seed(0)


g = 9.81 # Gravity
A = 8192 # Phillips spectrum parameter
WATER_ALBEDO = torch.tensor([0.00, 0.48, 0.57])
UP_DIRECTION = torch.tensor([0.00, 1.00, 0.00])


@dataclass
class OceanPatch:
    """ M x N resolution ocean patch of size Lx x Lz (m)
    """
    # Height and normal maps of the ocean patch
    height: Float32[Tensor, "M N"]
    normal: Float32[Tensor, "M N 3"]
    points: Float32[Tensor, "M N 3"]

    # Ocean patch metadata
    metadata: dict = field(default_factory=dict)

    def camera_bundle(self, fov: float = 90) -> RayBundle:
        """ Constructs camera ray bundle of a camera looking at center of ocean patch from directly above 
        """
        Lx = self.metadata['Lx']
        Lz = self.metadata['Lz']

        R = 2 * np.tan(np.radians(fov) / 2)
        W = self.metadata['M']
        H = self.metadata['N']
        fx = W / R
        fz = H / R
        cx = W / 2
        cz = H / 2
        camera_to_world = torch.tensor([ # nerfstudio uses [x, y, -z] convention
            [1, 0, 0, Lx / 2],
            [0, 0, 1, Lz / R], # altitude of camera
            [0, 1, 0, Lz / 2],
        ])
        camera = Cameras(camera_to_worlds=camera_to_world, fx=fx, fy=fz, cx=cx, cy=cz, height=H, width=W)
        return camera.generate_rays(torch.tensor([0])[:, None])


def generate_ocean_patch_wave_spectrum(
    k: Float[Tensor, "M N"],
    kx: Float[Tensor, "M"],
    kz: Float[Tensor, "N"],
    wind: Float[Tensor, "2"], 
    wind_alignment: float = 6,
    wave_dampening: float = 0.1,
) -> Float[Tensor, "M N"]:
    """ Generate an fourier spectrum of ocean waves acording to the Phillips spectrum

        P_h(k) = mean( |h_tilde*(k, t)^2| )
               = A * exp(-1 / (k * L)^2) / k^4 * cos(theta)^wave_alignment
            
    where wave_alignment measures alignment with wind direction and L (m) is the max wave length under wind speed. 
    Optionally, can supress small waves by a factor of 
    
        dampening = exp(-k^2 * wave_dampening^2)

    where wave_dampening is a small wave supression factor (m).

    Params:
        k: Wave numbers
        kx: Wave numbers in x direction
        kz: Wave numbers in y direction
        wind: Vector representing the wind direction and speed (m/s)
        wind_alignment: Power of cos(theta) in the Phillips spectrum
        wave_dampening: Small wave supression factor

    Returns:
        Wave spectrum of the ocean patch
    """
    wind_mag = torch.norm(wind)
    wind_dir = wind / wind_mag
    wave_max = wind_mag ** 2 / g # L parameter (m)
    cos_theta = (
        kx * wind_dir[0] + 
        kz * wind_dir[1]
    ) / k
    dampening = torch.exp(-k ** 2 * wave_dampening ** 2)
    wave_spectrum = (A * torch.exp(-1 / (k * wave_max) ** 2) / k ** 4) * (cos_theta ** wind_alignment) * dampening # P_h(k)
    wave_spectrum[k == 0] = 0 # Avoid division by zero
    return wave_spectrum


def generate_ocean_patch_dispersion_relation(k: Float[Tensor, "M N"], depth: Float[Tensor, "M N"] = None) -> Float[Tensor, "M N"]:
    """ Models the dispersion relation (frequency) of ocean waves in deep and shallow water: 
    In deep water, the dispersion relation is given by
        
        omega^2(k) = g * k
        
    For depth of D, the dispersion relation is given by
    
        omega^2(k) = g * k * tanh(k * D)
    """
    if depth is None:
        return torch.sqrt(g * k)
    return torch.sqrt(g * k * torch.tanh(k * depth))


def generate_ocean_patch(Lx: float, Lz: float, M: int, N: int, wind: Float[Tensor,"2"], t: float = 0, **kwargs) -> OceanPatch:
    """ Generate a random ocean patch at a given timestamp using FFT method described in:

    https://people.computing.clemson.edu/~jtessen/reports/papers_files/coursenotes2004.pdf (Section 4.3 and 4.4)

    The wave height of an Lx x Lz ocean patch at M x N resolution is given the 2D Discrete Fourier Transform

        h(x, t) = sum_k( h_k(t) * exp(i * k * x) )

    where k is the wave number vector, determined by the ocean patch size and grid resolution. 
    The slope vector is given by the gradient of the wave height

        e(x, t) = grad(h(x, t))
                = sum_k( i * k * h_k(t) * exp(i * k * x) )
    
    And the normal vector is given by
                
        n(x, t) = y_hat - e(x, t) / sqrt(1 + |e(x, t)|^2)

    Statisitcal analysis of waves have shown the fourier amplitudes h(k, t) can be modeled by gaussian random variables
    according to the wave spectrum P_h(k) and dispersion relation omega(k)

        h_(k, 0) = (N(0, 1) + i * N(0, 1)) * sqrt(P_h(k) / 2)
    
    and at time t

        h_(k, t) = h_(k, 0) * exp(i * omega(k) * t)
    
    Note the paper adds an additional term to preserve Hermitian symmetry in the fourier amplitudes

        + conj(h_(k, 0)) * exp(-i * omega(k) * t)

    but for our implementation, we can just use the irfft2 function.
        
    Params:
        Lx: Size of the ocean patch in x direction (m)
        Lz: Size of the ocean patch in y direction (m)
        M: Resolution of the ocean patch in x direction
        N: Resolution of the ocean patch in y direction
        t: Timestamp to simulate
        wind: Vector representing the wind direction and speed (m/s)

    Returns:
        Ocean patch with height and normal maps as well as position map for each grid point
    """
    # Generate wave numbers
    dx = 2 * torch.pi / Lx
    dy = 2 * torch.pi / Lz
    kx = torch.fft.fftfreq (M, d=dx) # Wave numbers in x (0, 1, ..., N/2, -N/2, ..., -1)
    kz = torch.fft.rfftfreq(N, d=dy) # Wave numbers in y (0, 1, ..., N/2)
    kx, kz = torch.meshgrid(kx, kz, indexing='xy')
    k = torch.sqrt(kx ** 2 + kz ** 2)

    # Compute Phillips spectrum
    P_h = generate_ocean_patch_wave_spectrum(k, kx, kz, wind, **kwargs)
    
    # Dispersion relation
    omega = generate_ocean_patch_dispersion_relation(k)
    
    # Fourier amplitudes at time 0 ~ gaussian random variables
    xi_real = torch.randn(N//2 + 1, M)
    xi_imag = torch.randn(N//2 + 1, M)
    h_k0 = (xi_real + 1j * xi_imag) * torch.sqrt(P_h) # real DC and Nyquist frequencies handled by irfft2

    # Fourier amplitudes at time t
    h_kt = h_k0 * torch.exp(1j * omega * t) #+ torch.conj(-h_k0) * torch.exp(-1j * omega * t)

    # Compute height, tangent, and normal maps
    hmap = torch.fft.irfft2(h_kt, s=(N, M))
    hmap_grad_x = torch.fft.irfft2(1j * kx * h_kt, s=(N, M)) * M / Lx # rescale to match grid resolution
    hmap_grad_z = torch.fft.irfft2(1j * kz * h_kt, s=(N, M)) * N / Lz
    hmap_grad = torch.stack([hmap_grad_x, hmap_grad_z], dim=-1)
    nmap = torch.stack([-hmap_grad_x, torch.ones_like(hmap), -hmap_grad_z], dim=-1)
    nmap = torch.nn.functional.normalize(nmap, p=2, dim=-1)

    # Compute positions at each grid point
    x = torch.linspace(0, Lx, M)
    z = torch.linspace(0, Lz, N)
    x, z = torch.meshgrid(x, z, indexing='ij')
    rmap = torch.stack([x, hmap, z], dim=-1)

    return OceanPatch(hmap, nmap, rmap, dict(Lx=Lx, Lz=Lz, M=M, N=N, t=t, wind=wind, height_grad=hmap_grad))


def points2indices(points: Float[Tensor, "H W 3"], M: int, N: int, Lx: float, Lz: float, mod=False) -> tuple[
    Float[Tensor, "H * W"],
    Float[Tensor, "H * W"]
]:
    """
    """
    x = torch.floor(points[..., 0] * M / Lx).long()
    z = torch.floor(points[..., 2] * N / Lz).long()
    if mod: # assuming consistent points, useful for accumulation operations for points that go off image edge
        x = x % M
        z = z % N
    mask = torch.logical_and(x >= 0, x < M) & \
           torch.logical_and(z >= 0, z < N)
    return x[mask], z[mask], mask


def generate_ocean_bottom_lightmap(
    patch: OceanPatch,
    image: Float[Tensor, "H W 3"],
    depth: Float[Tensor, "H W"] = None,
    light: Float[Tensor, "3"] = None,
    light_ambient: float = 0,
    light_scatter: float = 0,
    device='cuda'
) -> Float[Tensor, "H W 3"]:
    """
    """
    H, W = image.shape[:2]
    M, N, Lx, Lz = patch.metadata['M'], patch.metadata['N'], patch.metadata['Lx'], patch.metadata['Lz']
    depth = depth if depth is not None else torch.full((H, W), 5.0)
    light = light if light is not None else -UP_DIRECTION
    image = image.to(device)
    depth = depth.to(device)
    light = light.to(device)

    ni = light / torch.norm(light.float())
    ns = patch.normal
    ni = ni.to(device)
    ns = ns.to(device)
    nt = refraction(ni, ns, REFRACTION_INDEX['water'], REFRACTION_INDEX['air'])
    _, transmission_mult = compute_fresnel(ni, ns, nt) # (H, W)

    # trace light rays from ocean surface to ocean bottom
    points = patch.points.to(device)
    distance = torch.abs((points[..., 1] + depth) / nt[..., 1]) # (H, W)
    bottom_points = points - distance[..., None] * nt # (H, W, 3)

    x, z, mask = points2indices(bottom_points, M, N, Lx, Lz)
    accum = torch.zeros_like(image)
    accum[x, z] += compute_transmission(
        image,
        distance[..., None],
        light,
        light_ambient, 
        light_scatter,
        transmission_mult[..., None]
    )[mask] * WATER_ALBEDO.to(device)
    
    # lightmap is density based so smooth operation should be average
    accum = accum.permute(2, 0, 1)[None]
    accum = torch.nn.functional.avg_pool2d(accum, 11, stride=1, padding=5)
    accum = accum[0].permute(1, 2, 0)
    
    return torch.clamp(accum, 0, 1)


def apply_corruption_ocean(
    patch: OceanPatch,
    image: Float[Tensor, "H W 3"],
    depth: Float[Tensor, "H W"] = None,
    light: Float[Tensor, "3"] = None,
    light_ambient: float = 0, 
    light_scatter: float = 0,
    light_specular_mult: float = 0.95,
    light_specular_gain: float = 0.1,
    device='cuda',
) -> Float[Tensor, "H W 3"]:
    """
    """
    H, W = image.shape[:2]
    M, N, Lx, Lz = patch.metadata['M'], patch.metadata['N'], patch.metadata['Lx'], patch.metadata['Lz']
    depth = depth if depth is not None else torch.full((H, W), 5)
    light = light if light is not None else -UP_DIRECTION
    image = image.to(device)
    depth = depth.to(device)
    light = light.to(device)
    image_bottom = generate_ocean_bottom_lightmap(
        patch, 
        image, 
        depth, 
        light, 
        light_ambient=light_ambient, 
        light_scatter=light_scatter, 
        device=device
    )

    # compute camera ray bundle
    bundle = patch.camera_bundle()

    # reflection and refraction are symmetric wrt time reversal
    ni = bundle.directions[..., 0, :]
    ns = patch.normal
    ni = ni.to(device)
    ns = ns.to(device)
    nr = reflection(ni, ns)
    nt = refraction(ni, ns, REFRACTION_INDEX['air'], REFRACTION_INDEX['water'])
    reflection_mult, transmission_mult = compute_fresnel(ni, ns, nt)

    accum = torch.zeros((H, W, 3), device=device)

    # trace (inverse) light rays from ocean surface to ocean bottom
    points = patch.points.to(device)
    distance = torch.abs((points[..., 1] + depth) / nt[..., 1]) # (H, W)
    bottom_points = points - distance[..., None] * nt  # (H, W, 3)

    # constant directional light so can accumulate off grid points
    x, z, _ = points2indices(bottom_points, M, N, Lx, Lz, mod=True)
    x = x.reshape(H, W)
    z = z.reshape(H, W)
    accum = compute_transmission(
        image_bottom,
        depth[..., None],
        light,
        light_ambient,
        light_scatter, 
        transmission_mult[..., None]
    )[x, z] * WATER_ALBEDO.to(device)
    #accum = torch.log(1 + accum) # smooth out lightmap

    # compute reflection by seeing if unreflected ray is within cosine threshold of vertical
    reflection_unit = torch.tensor([light[0], -light[1], light[2]], device=device) / torch.norm(light)
    reflection_mask = torch.sum(nr * reflection_unit, dim=-1) > light_specular_mult
    reflection_gain = torch.norm(light) * reflection_mult[..., None] * light_specular_gain
    #reflection_gain = reflection_gain * WATER_ALBEDO.to(device)
    accum = torch.where(reflection_mask[..., None], reflection_gain + accum, accum)

    return torch.clamp(accum, 0, 1)


if __name__ == '__main__':
    pass