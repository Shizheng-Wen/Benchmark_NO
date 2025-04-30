import torch 
from dataclasses import dataclass
from typing import Union, Tuple

def rescale(x: torch.Tensor, lims=(-1,1), phys_domain = ([-1, -1], [1, 1])):
    min_vals = torch.tensor(phys_domain).min()
    max_vals = torch.tensor(phys_domain).max()
    
    rescaled = ((x - min_vals) / (max_vals - min_vals)) * (lims[1] - lims[0]) + lims[0]

    return rescaled