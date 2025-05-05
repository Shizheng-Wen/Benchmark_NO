import torch
from typing import Union, Tuple

def remove_duplicate_edges(edges:torch.Tensor,return_idx:bool = False
    )->Union[torch.Tensor,Tuple[torch.Tensor,torch.Tensor]]:
    """

    Parameters
    ----------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]

    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    # NOTE: edges should be sorted first
    assert edges.ndim == 2 and edges.shape[0] == 2, f"The edges are expected to be [2, n_edges], but got shape {edges.shape}"
    assert edges.min() >= 0, "The edges are expected to be non-negative"
    assert edges.max() < torch.iinfo(torch.int32).max, "The edges are expected to be less than the maximum integer value"
    dtype = edges.dtype
    edges_id = edges[0].type(torch.int64) << 32 | edges[1].type(torch.int64)

    if return_idx:
        unique_edges_id, counts = edges_id.unique(return_counts=True) # [n_unique_edges]
        unique_idx = torch.cumsum(counts, 0) - counts
    else:
        unique_edges_id = edges_id.unique() # [n_unique_edges]
    unique_edges = torch.stack([
        unique_edges_id >> 32,
        unique_edges_id & 0xFFFFFFFF
    ], 0)
    unique_edges = unique_edges.type(dtype)
    if return_idx:
        return unique_edges, unique_idx
    else:
        return unique_edges

def sort_edges(edges:torch.Tensor, return_idx:bool=False
               )->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Parameters
    ----------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    return_idx: bool
        whether to return the indices

    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    or 
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    idx: torch.Tensor
        1D tensor of shape [n_edges]
    """
    assert edges.ndim == 2 and edges.shape[0] == 2, f"The edges are expected to be [2, n_edges], but got shape {edges.shape}"
    assert edges.min() >= 0, "The edges are expected to be non-negative"
    assert edges.max() < torch.iinfo(torch.int32).max, "The edges are expected to be less than the maximum integer value"
    dtype = edges.dtype
    edges_id = edges[0].type(torch.int64) << 32 | edges[1].type(torch.int64)
    sorted_edges_id = edges_id.argsort()
    if return_idx:
        return edges[:, sorted_edges_id].type(dtype), sorted_edges_id   
    else:
        return edges[:, sorted_edges_id].type(dtype)
