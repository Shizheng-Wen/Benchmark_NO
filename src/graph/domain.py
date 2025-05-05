import torch
from typing import Sequence
from functools import lru_cache

@lru_cache()
def domain_shifts(span:Sequence[int]=(2,2))->torch.Tensor:
    """
    Parameters
    ----------
    span: Sequence[int] 
        span of domains in each dimension

    Returns
    -------
    domain_shifts: torch.Tensor
        2D tensor of shape [3^n_dim-1, n_dim]
    """
    values = []
    for s in span:
        values.append(torch.tensor([0, -s, s]))
    
    grids  = torch.meshgrid(*values, indexing='ij')
    points = torch.stack(grids, dim=-1).reshape(-1, len(span))
    
    return points

def shift(points:torch.Tensor, domain_shifts:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    points:torch.Tensor
        2D Tensor of shape [n_points, n_dimension]
    domain_shifts:torch.Tensor 
        2D Tensor of shape [n_shifts, n_dimension]

    Returns
    -------
    points:torch.Tensor
        2D Tensor of shape [n_points*n_shifts, n_dimension]
    """
    assert points.ndim == 2, f"The points are expected to be 2D tensor, but got shape {points.shape}"
    assert domain_shifts.ndim == 2, f"The domain_shifts are expected to be 2D tensor, but got shape {domain_shifts.shape}"
    assert points.shape[-1] == domain_shifts.shape[-1], \
        f"The points and domain_shifts should have the same number of dimensions, but got {points.shape[-1]} and {domain_shifts.shape[-1]}"
    n_dimension = points.shape[-1]
    points = points[None, :, :] + domain_shifts[:, None, :]
    return points.reshape(-1, n_dimension)
