import torch 
from typing import Optional
from .tri import delaunay, tri_medians
from .domain import shift

def minimal_support(points:torch.Tensor, 
                    domain_shifts:Optional[torch.Tensor] = None
                    )->torch.Tensor:
    """
    compute the maximal median support of a set of points. These medians are computed 
    by the Delaunay triangulation of the points.

    Parameters
    ----------
    points: torch.Tensor
        2D Tensor of shape [n_points, n_dim]
    domain_shifts: torch.Tensor
        2D Tensor of shape [n_shifts, n_dim]

    Returns
    -------
    radii: torch.Tensor
        1D tensor of shape [n_points,]
    """

    if domain_shifts is not None:
        shifted_points = shift(points, domain_shifts)
        triangles = delaunay(shifted_points)
        medians   = tri_medians(shifted_points[triangles])
    else:
        triangles = delaunay(points)
        medians   = tri_medians(points[triangles])
    
    mask = triangles < points.shape[0] # [N, 3] in case of shifted
    values = medians[mask]
    indices  = triangles[mask]
    radii = torch.full((points.shape[0],), -1, dtype=points.dtype, device=points.device)
    radii = torch.index_reduce(radii, 0, indices, values, reduce="amax")

    return radii
