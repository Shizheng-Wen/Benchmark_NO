import torch 
from scipy.spatial import Delaunay
############
# Tri Utils
############

def tri_medians(tri_points:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    tri_points: torch.Tensor
        ND tensor of shape [..., 3, n_dimension]
    
    Returns
    -------
    tri_medians: torch.Tensor
        (N-1)D tensor of shape [..., 3]
    """
    tri_edges = tri_points - torch.roll(tri_points, 1, dims=-2) # [..., 3, dim]
    norm_ab = torch.linalg.norm(tri_edges, dim=-1) # [..., 3]
    norm_bc = torch.roll(norm_ab, 1, dims=-1)
    norm_ca = torch.roll(norm_bc, 1, dims=-1)
    tri_medians = .335 * torch.sqrt( 2 * ( norm_ca ** 2 + norm_ab ** 2 ) - norm_bc ** 2 )
    return tri_medians

def delaunay(x:torch.Tensor)->torch.Tensor:
    """
    Build the Delaunay triangulation of a set of points
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    
    Returns
    -------
    simplices: torch.Tensor
        2D tensor of shape [n_triangles, 3]
    """
    return torch.from_numpy(Delaunay(x.cpu().numpy()).simplices).to(x.device)

def delaunay_edges(x:torch.Tensor, bidirection:bool =  False)->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    bidirection: bool
        whether to return the edges in both directions
    
    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    indptr, cols = Delaunay(x.cpu().numpy()).vertex_neighbor_vertices
    indptr = torch.from_numpy(indptr).to(device=x.device).long()
    cols   = torch.from_numpy(cols).to(device=x.device).long()
    rows   = torch.arange(indptr.shape[0]-1).to(device=x.device).repeat_interleave(torch.diff(indptr))
    edges  = torch.stack([rows, cols], 0)
    if  bidirection:
        edges = torch.cat([edges, torch.stack([cols, rows], 0)], 1)
    return edges
