import torch
from torch import Tensor
from jaxtyping import Float, Bool


def points2indices(points: Float[Tensor, "H W 3"], M: int, N: int, Lx: float, Lz: float, mod=False) -> tuple[
    Float[Tensor, "H W"],
    Float[Tensor, "H W"],
]:
    """
    """
    xi = torch.floor(points[..., 0] * N / Lx).int()
    zi = torch.floor(points[..., 2] * M / Lz).int()
    if mod: # assuming consistent points, useful for accumulation operations for points that go off image edge
        xi %= N
        zi %= M
    mask = torch.logical_and(xi >= 0, xi < N) & \
           torch.logical_and(zi >= 0, zi < M)
    indices = torch.stack([xi, zi], dim=-1)
    return indices, mask


def trace_points(
    points    : Float[Tensor, "H W 3"], 
    directions: Float[Tensor, "H W 3"], 
    depth     : Float[Tensor, "H W"], 
    Lx: float, 
    Lz: float,
    half_precision=True, 
    nbatches=1
) -> tuple[
    Float[Tensor, "H W 3"],
    Float[Tensor, "H W"],
    Float[Tensor, "H W"],   
]:
    """
    """
    H, W, _ = points.shape
    distance_upper_bound = -(points[..., 1] + depth.max()) / directions[..., 1]
    tdelta = min(Lx / W, Lz / H) # grid cell step size lower bound
    nsteps = torch.ceil(distance_upper_bound.max() / tdelta).long().item()

    if half_precision:
        points, directions, depth = \
            points.to(torch.float16), directions.to(torch.float16), depth.to(torch.float16)

    def push(steps: Float[Tensor, "T"]) -> tuple[
        Float[Tensor, "H W"],
        Bool [Tensor, "H W"],
    ]:
        """
        """
        path = points[..., None] + directions[..., None] * steps.to(points) * tdelta # (H, W, 3, nsteps)
        path = path.permute(0, 1, 3, 2)
        path_depth = -path[..., 1] # (H, W, nsteps)

        indices, _ = points2indices(path, H, W, Lx, Lz, mod=True) # tile depth patch via mod=True
        x = indices[..., 0]
        z = indices[..., 1]
        path_depth_bottom = depth[z, x] # (H, W, nsteps)
        intersect = (path_depth >= path_depth_bottom).int() # (H, W, nsteps)
        intersect_mask = torch.sum(intersect, dim=-1) > 0 # (H, W)

        t = torch.argmax(intersect, dim=-1) # (H, W)
        intersect_depth = path_depth_bottom[
            torch.arange(H)[:, None],
            torch.arange(W)[None, :],
            t
        ].float() # (H, W)
        return intersect_depth, intersect_mask

    batch_step = nsteps // nbatches + (nsteps % nbatches > 0)
    
    intersect_depth = torch.full((H, W),    -1, device=points.device)
    intersect_mask  = torch.full((H, W), False, device=points.device)
    for b in range(nbatches):
        i = b * batch_step
        j = i + batch_step
        steps = torch.arange(i, min(j, nsteps))
        intersect_depth_batch, intersect_mask_batch = push(steps)
        intersect_depth = torch.where(~intersect_mask, intersect_depth_batch, intersect_depth)
        intersect_mask |= intersect_mask_batch
    
    intersect_depth = torch.where(~intersect_mask, depth.max(), intersect_depth) # catch all case

    distance = -(points[..., 1] + intersect_depth) / directions[..., 1]
    bottom_points = points + distance[..., None] * directions # (H, W, 3)

    assert torch.allclose(bottom_points[..., 1], -intersect_depth, atol=1e-5)
    return bottom_points, distance