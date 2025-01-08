import multiprocessing as mp

import torch
from torch import Tensor
from jaxtyping import Float, Bool
from scipy.ndimage import distance_transform_edt


REFRACTION_INDEX = {'air': 1.0, 'water': 1.33}


def reflection(
    ni: Float[Tensor, "... 3"], 
    ns: Float[Tensor, "... 3"]
) -> Float[Tensor, "... 3"]:
    """ Compute the reflection of a vector ni with respect to a normal ns
    """
    return ni - 2 * torch.sum(ni * ns, dim=-1, keepdims=True) * ns


def refraction(
    ni: Float[Tensor, "... 3"], 
    ns: Float[Tensor, "... 3"], 
    eta1: float, 
    eta2: float
) -> Float[Tensor, "... 3"]:
    """ Compute the refraction of a vector ni with respect to a normal ns given the refractive indices n1 and n2.
        Handles total internal reflection, which sets refraction to zero.
    """
    eta = eta1 / eta2
    cos_theta_i = torch.abs(torch.sum(ni * ns, dim=-1, keepdims=True))
    sin_theta_t = torch.sqrt(1.0 - cos_theta_i ** 2) * eta
    cos_theta_t = torch.sqrt(1.0 - sin_theta_t ** 2)

    total_internal_reflection = sin_theta_t ** 2 > 1.0
    nt = eta * ni + (eta * cos_theta_i - cos_theta_t) * ns
    nt = torch.where(total_internal_reflection, torch.zeros_like(nt), nt)
    return nt


def compute_fresnel(
    ni: Float[Tensor, "... 3"], 
    ns: Float[Tensor, "... 3"], 
    nt: Float[Tensor, "... 3"]
) -> tuple[float, float]:
    """ Compute fresnel reflection and transmission coefficients given the incident and outgoing angles 
    where we assume no loss of energy i.e. R + T = 1
    """
    theta_i = torch.acos(torch.abs(torch.sum(ni * ns, dim=-1)))
    theta_t = torch.acos(torch.abs(torch.sum(nt * ns, dim=-1)))
    R = 0.5 * (
        torch.sin(theta_t - theta_i) ** 2 / (torch.sin(theta_t + theta_i) + 1e-6) ** 2 + \
        torch.tan(theta_t - theta_i) ** 2 / (torch.tan(theta_t + theta_i) + 1e-6) ** 2
    )
    # handle total internal reflection
    R = torch.where(nt.sum(dim=-1) == 0, torch.ones_like(R), R)
    return R, 1 - R


def compute_transmission(
    image: Float[Tensor, "... 3"],
    depth: Float[Tensor, "... 1"],
    light_ambient: float,
    light_scatter: float,
) -> Float[Tensor, "... 3"]:
    """ Compute the transmission of light through a medium according to Koschmieder's law.
    """
    transmission_map = torch.exp(-light_scatter * depth)
    return transmission_map * image + light_ambient * (1 - transmission_map)


if __name__ == '__main__':
    # test reflection
    ni = torch.tensor([ 1, -1,  0]).float()
    ns = torch.tensor([ 0,  1,  0]).float()
    ni = ni / torch.norm(ni)
    ns = ns / torch.norm(ns)
    print(reflection(ni, ns)) # [1, 1, 0] / norm([1, 1, 0]) = [0.7071, 0.7071, 0]

    # test refraction
    ni = torch.tensor([ 1, -1,  0]).float()
    ns = torch.tensor([ 0,  1,  0]).float()
    ni = ni / torch.norm(ni)
    ns = ns / torch.norm(ns)
    print(refraction(ni, ns, 1.00, 1.33)) # [0.530, -0.848, 0]

    ni = torch.tensor([ 9, -1,  0]).float() # total internal reflection
    ns = torch.tensor([ 0,  1,  0]).float()
    ni = ni / torch.norm(ni)
    ns = ns / torch.norm(ns)
    print(refraction(ni, ns, 1.33, 1.00)) # [0, 0, 0]

    # test fresnel
    ni = torch.tensor([ 1, -1,  0]).float()
    ns = torch.tensor([ 0,  1,  0]).float()
    ni = ni / torch.norm(ni)
    ns = ns / torch.norm(ns)
    nt = refraction(ni, ns, 1.00, 1.33)
    print(compute_fresnel(ni, ns, nt)) # (0.028, 0.972)

    # test interpolation
    tensor = torch.tensor([[[1, 2, 3], [0, 0, 0], [7, 8, 9]]]).float()
    mask = torch.tensor([[0, 1, 0]]).bool()
    print(interpolate_with_nearest(tensor, mask)) # [[[1, 2, 3], [1, 2, 3], [7, 8, 9]]]