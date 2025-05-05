import torch 
import numpy as np
from typing import Optional


#################
# Encoding Utils
#################
def node_pos_encode(x:torch.Tensor,
                    ndata:Optional[torch.Tensor]=None,
                    freq:int = 4,
                    periodic:bool = False,
                    add_dummy_node:bool = False
               )->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    ndata: Optional[torch.Tensor]
        2D tensor of shape [n_points, n_features] or None
    freq: int
        number of frequencies
    periodic: bool
        whether the encoding is periodic
    add_dummy_node: bool
        whether to add a dummy node

    Returns
    -------
    x_encoded: torch.Tensor
        2D tensor of shape [n_points or n_points+1, 2*freq or 2*freq+n_features or n_dimension or n_dimension+1]
    """
    assert x.ndim == 2, f"The x is expected to be 2D tensor, but got shape {x.shape}"
    assert ndata is None or (ndata.ndim == 2 and ndata.shape[0] == x.shape[0]), f"The ndata is expected to be 2D tensor, but got shape {ndata.shape}"
    if periodic:
        device = x.device
        freqs = torch.arange(1, freq+1).to(device=device) # [freq]
        phi   = np.pi * (x + 1)
        x = freqs[None, :, None] * phi[:, None, :] # [n_points, 1, dim] * [1, freq, 1] -> [n_points, freq, dim]
        x = torch.cat([x.sin(), x.cos()], dim=2)  # [n_points, freq, dim * 2]
        x = x.view(x.shape[0], -1)  # [n_points, freq * 2 * dim]
    
    if ndata is not None:
        dtype = x.dtype
        x = x.type(dtype)
        x = torch.cat([x, ndata], axis=-1)
 
    
    if add_dummy_node:
        x = add_dummy(x)
    
    return x

def edge_pos_encode(u:torch.Tensor,
                    v:torch.Tensor,
                    edges:torch.Tensor,
                    edata:Optional[torch.Tensor] = None,
                    periodic:bool = False,
                    max_edge_length:float = 2.0,
                    domain_shifts:Optional[torch.Tensor] = None,
                    domain_edges:Optional[torch.Tensor] = None
                    )->torch.Tensor:
    """
    Parameters
    ----------
    u: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    v: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    edata: Optional[torch.Tensor]
        2D tensor of shape [n_edges, n_features]
    periodic: bool
        whether the encoding is periodic
    max_edge_length: float
        maximum edge length
    domain_shifts: Optional[torch.Tensor]
        2D tensor of shape [n_shifts, n_dimension]
    domain_edges: Optional[torch.Tensor]
        2D tensor of shape [2, n_domain]

    Returns
    -------
    edata: torch.Tensor
        2D tensor of shape [n_edges, 2*n_dimension+1]
    """

    u_e, v_e = u[edges[0]], v[edges[1]] # [n_edges, n_dim]
    z_uv = u_e - v_e # [n_edges, n_dim]
    assert (z_uv.abs() <= 2.0).all(), "The edge length should be less than 2.0"
    if periodic:
        if domain_shifts is None:
            z_uv = torch.where(z_uv >= 1., z_uv - 2., z_uv)
            z_uv = torch.where(z_uv < -1., z_uv + 2., z_uv)
        else:
            z_uv = ( u[edges[0]] + domain_shifts[domain_edges[0]] ) \
                 - ( v[edges[1]] + domain_shifts[domain_edges[1]] ) # [n_edges, n_dim]
        
    d_uv = torch.linalg.norm(z_uv, axis=-1, keepdims=True) # [n_edges, 1]

    assert (z_uv.abs() <= max_edge_length).all(), z_uv.abs().max()
    assert (d_uv.abs() <= max_edge_length).all(), d_uv.abs().max()

    z_uv = z_uv / max_edge_length
    d_uv = d_uv / max_edge_length
    edata_struc = torch.cat([z_uv, d_uv], axis=-1) # [n_edges, 2*n_dim+1]
    if edata is not None:
        edata = torch.cat([edata_struc, edata], axis=-1)
    return edata_struc

def add_dummy(x:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor of shape [..., n_points, n_features]

    Returns
    -------
    x: torch.Tensor
        ND tensor of shape [..., n_points+1, n_features]
    """
    assert x.ndim >= 2, f"The x is expected to be ND tensor, but got shape {x.shape}"
    device = x.device
    dtype  = x.dtype
    shape  = list(x.shape)
    shape[-2] = 1
    x = torch.cat([x, torch.zeros(shape, device=device, dtype=dtype)], -2)
    return x

def remove_dummy(x:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [..., n_points, n_features]

    Returns
    -------
    x: torch.Tensor
        2D tensor of shape [..., n_points-1, n_features]
    """
    assert x.ndim >= 2, f"The x is expected to be ND tensor, but got shape {x.shape}"
    return x[..., :-1, :]
 
 