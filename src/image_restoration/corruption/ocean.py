from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from jaxtyping import Float
from nerfstudio.cameras.cameras import Cameras, RayBundle

from image_restoration.corruption.optics import *
from image_restoration.corruption.geom import *


np.random.seed(0)


g = 9.81 # Gravity
UP_DIRECTION = torch.tensor([0.00, 1.00, 0.00]) # camera angle
WATER_ALBEDO_TOP = torch.tensor([0.60, 0.60, 0.60])
WATER_ALBEDO_BOT = torch.tensor([0.00, 0.48, 0.57])
WATER_ALBEDO_END = 1


@dataclass
class OceanPatch:
    """ N x M resolution ocean patch of size Lz x Lx (m)
    """
    # Height and normal maps of the ocean patch
    height: Float[Tensor, "N M"]
    normal: Float[Tensor, "N M 3"]
    points: Float[Tensor, "N M 3"]

    # Ocean patch metadata
    metadata: dict = field(default_factory=dict)

    def camera_bundle(self, fov: float = 60) -> RayBundle:
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
    wind_alignment: float = 6, # 0 to 10
    wave_dampening: float = 0.1, # 
    wave_amplitude: float = 4096, # scaling parameter
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
    wave_spectrum = (wave_amplitude * torch.exp(-1 / (k * wave_max) ** 2) / k ** 4) * (cos_theta ** wind_alignment) * dampening # P_h(k)
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


def generate_ocean_patch(Lx: float, Lz: float, M: int, N: int, wind: Float[Tensor,"2"], t = 0, **kwargs) -> OceanPatch:
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
        M: Resolution of the ocean patch in x direction (cols)
        N: Resolution of the ocean patch in z direction (rows)
        t: Timestamp to simulate
        wind: Vector representing the wind direction and speed (m/s)

    Returns:
        Ocean patch with height and normal maps as well as position map for each grid point
    """
    # Generate wave numbers
    dx = 2 * torch.pi / Lx * M
    dz = 2 * torch.pi / Lz * N
    kx = torch.fft.fftfreq (M) * dx # Wave numbers in x (0, 1, ..., N/2, -N/2, ..., -1)
    kz = torch.fft.rfftfreq(N) * dz # Wave numbers in y (0, 1, ..., N/2)
    kx, kz = torch.meshgrid(kx, kz, indexing='xy')
    k = torch.sqrt(kx ** 2 + kz ** 2)

    # Compute Phillips spectrum
    P_h = generate_ocean_patch_wave_spectrum(k, kx, kz, wind, **kwargs)
    
    # Dispersion relation
    omega = generate_ocean_patch_dispersion_relation(k)
    
    # Fourier amplitudes at time 0 ~ gaussian random variables
    xi_real = torch.randn(N // 2 + 1, M)
    xi_imag = torch.randn(N // 2 + 1, M)
    h_k0 = (xi_real + 1j * xi_imag) * torch.sqrt(P_h) # real DC and Nyquist frequencies handled by irfft2

    # Fourier amplitudes at time t
    h_kt = h_k0 * torch.exp(1j * omega * t) #+ torch.conj(-h_k0) * torch.exp(-1j * omega * t)

    # Compute height, tangent, and normal maps
    scale = (M * N) / (Lx * Lz)
    hmap = scale * torch.fft.irfft2(h_kt, s=(N, M))
    hmap_grad_x = scale * torch.fft.irfft2(1j * kx * h_kt, s=(N, M))
    hmap_grad_z = scale * torch.fft.irfft2(1j * kz * h_kt, s=(N, M))
    hmap_grad = torch.stack([hmap_grad_x, hmap_grad_z], dim=-1)
    nmap = torch.stack([-hmap_grad_x, torch.ones_like(hmap), -hmap_grad_z], dim=-1)
    nmap = torch.nn.functional.normalize(nmap, dim=-1)

    # Compute positions at each grid point
    x = torch.linspace(0, Lx, M + 1)[:-1]
    z = torch.linspace(0, Lz, N + 1)[:-1]
    x, z = torch.meshgrid(x, z, indexing='xy')
    rmap = torch.stack([x, hmap, z], dim=-1)

    return OceanPatch(hmap, nmap, rmap, dict(Lx=Lx, Lz=Lz, M=M, N=N, t=t, wind=wind, height_grad=hmap_grad))


def generate_ocean_albedo(depth: Float[Tensor, "H W"]) -> Float[Tensor, "H W 3"]:
    """ Generate a random ocean albedo map
    """
    albedo_top = WATER_ALBEDO_TOP.to(depth.device)
    albedo_bot = WATER_ALBEDO_BOT.to(depth.device)
    return torch.where(
        depth[..., None] < WATER_ALBEDO_END, 
        albedo_top + (albedo_bot - albedo_top) * depth[..., None], 
        albedo_bot
    )
    #return WATER_ALBEDO.to(depth.device).expand(depth.shape[0], depth.shape[1], 3)


def generate_ocean_bottom_lightmap(
    patch: OceanPatch,
    image: Float[Tensor, "H W 3"],
    depth: Float[Tensor, "H W"],
    light: Float[Tensor, "3"], # light coming from the sun
    light_ambient: float = 0, # light already in the ocean
    light_scatter: float = 0, # "fog" equivalent, light attenuation
    device='cuda'
) -> Float[Tensor, "H W 3"]:
    """
    """
    H, W = image.shape[:2]
    M, N, Lx, Lz = patch.metadata['M'], patch.metadata['N'], patch.metadata['Lx'], patch.metadata['Lz']
    image = image.to(device)
    depth = depth.to(device)
    light = light.to(device)

    ni = light / torch.norm(light.float())
    ns = patch.normal
    ni = ni.to(device)
    ns = ns.to(device)
    nt = refraction(ni, ns, REFRACTION_INDEX['air'], REFRACTION_INDEX['water'])
    _, transmission_mult = compute_fresnel(ni, ns, nt) # (H, W)

    # trace light rays from ocean surface to ocean bottom
    points = patch.points.to(device)
    bottom_points, distance = trace_points(points, nt, depth, Lx, Lz)

    # compute surface light and albedo maps
    lightmap = torch.full_like(image, torch.norm(light)) * transmission_mult[..., None]
    colormap = generate_ocean_albedo(distance)

    # accumulate bottom lightmap
    indices, _ = points2indices(bottom_points, M, N, Lx, Lz, mod=True) # tile ocean patch via mod=True
    x = indices[..., 0]
    z = indices[..., 1]
    accum = torch.zeros_like(image)
    accum[z, x] += compute_transmission(
        lightmap,
        distance[..., None],
        light_ambient,
        light_scatter,
    ) * colormap

    # smooth lightmap
    accum = accum.permute(2, 0, 1)[None]
    accum = torch.nn.functional.max_pool2d(accum, 5, stride=1, padding=2)
    accum = accum[0].permute(1, 2, 0)
    
    # add texture
    accum = accum * image

    return torch.clamp(accum, 0, 1)


def apply_corruption_ocean(
    patch: OceanPatch,
    image: Float[Tensor, "H W 3"],
    depth: Float[Tensor, "H W"],
    light: Float[Tensor, "3"],
    light_ambient: float = 0, 
    light_scatter: float = 0,
    light_specular_gain: float = 0.1, # how powerful the reflecting source is
    light_specular_gpow: float = 1.0, # how much the specular reflection is focused
    device='cuda',
) -> Float[Tensor, "H W 3"]:
    """
    """
    H, W = image.shape[:2]
    M, N, Lx, Lz = patch.metadata['M'], patch.metadata['N'], patch.metadata['Lx'], patch.metadata['Lz']
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
    ni = bundle.directions[..., 0, :].float()
    ns = patch.normal
    ni = ni.to(device)
    ns = ns.to(device)
    nr = reflection(ni, ns)
    nt = refraction(ni, ns, REFRACTION_INDEX['air'], REFRACTION_INDEX['water'])
    reflection_mult, transmission_mult = compute_fresnel(ni, ns, nt)

    # trace (inverse) light rays from ocean surface to ocean bottom assuming constant depth
    points = patch.points.to(device)
    bottom_points, distance = trace_points(points, nt, depth, Lx, Lz)

    # compute ocean surface transmission and albedo maps
    imagemap = image_bottom * transmission_mult[..., None]
    colormap = generate_ocean_albedo(distance)

    # accumlate camera image
    indices, mask = points2indices(bottom_points, M, N, Lx, Lz, mod=True)
    x = indices[..., 0]
    z = indices[..., 1]
    imagemap[~mask] = 0 # set invalid points color to 0
    accum = compute_transmission(
        imagemap,
        distance[..., None],
        light_ambient,
        light_scatter,
    )[z, x] * colormap[z, x]
    #accum = torch.log(1 + accum) # smooth out lightmap

    # reflection is symmetric wrt time reversal
    reflection_unit = torch.tensor([-light[0], -light[1], light[2]], device=device) / torch.norm(light)
    reflection_dotp = torch.sum(nr * reflection_unit, dim=-1)[..., None]
    reflection_gain = torch.norm(light) * reflection_mult[..., None] * (reflection_dotp ** light_specular_gpow) * light_specular_gain
    accum = reflection_gain + accum

    return torch.clamp(accum, 0, 1)


if __name__ == '__main__':
    pass