import numpy as np
import torch 
import torch.nn as nn 
import numpy as np
from typing import  Optional
from functools import partial
from .graph import Graph, radius_bipartite_graph, hierarchical_graph
from .domain import domain_shifts
from .support import minimal_support
from src.utils.sample import subsample, grid
from src.utils.scale import rescale

class RegionInteractionGraph(nn.Module):
    physical_to_regional:Graph # bipartite
    regional_to_regional:Graph # homogeneous
    regional_to_physical:Graph # bipartite

    def __init__(self, 
                 physical_to_regional:Graph,
                 regional_to_regional:Graph,
                 regional_to_physical:Graph):
        super().__init__()
        self.physical_to_regional = physical_to_regional
        self.regional_to_regional = regional_to_regional
        self.regional_to_physical = regional_to_physical

    @classmethod
    def from_point_cloud(cls, points: torch.Tensor,
                         output_points: Optional[torch.Tensor] = None,
                         phy_domain: Optional[tuple] = None,
                         periodic: bool = False,
                         sample_factor: float = 0.5,
                         overlap_factor_p2r: float = 1.0,
                         overlap_factor_r2p: float = 1.0,
                         regional_level: int = 1,
                         add_dummy_node: bool = False,
                         with_additional_info:bool = True,
                         regional_points: Optional[list] = None,
                         fix_radius: Optional[float] = None,
                         **kwargs):
        
        physical_points = rescale(points, (-1, 1))  
        if output_points is not None:
            output_physical_points = rescale(output_points, (-1, 1))
        else:
            output_physical_points = physical_points
        if regional_points is None:
            if periodic:
                _domain_shifts = domain_shifts((2.0,2.0)).to(physical_points.device)
                if sample_factor <= 1.0:
                    regional_points = subsample(physical_points, factor=sample_factor)
                else:
                    regional_points = subsample(physical_points, n=int(sample_factor))
            else:
                _domain_shifts = None
                if sample_factor <= 1.0:
                    regional_points = subsample(physical_points, factor=sample_factor)
                else:
                    regional_points = subsample(physical_points, n=int(sample_factor))
        else:
            x_min, y_min = phy_domain[0]
            x_max, y_max = phy_domain[1]
            assert x_min < x_max and y_min < y_max, "Invalid physical domain"
            meshgrid = torch.meshgrid(torch.linspace(x_min, x_max, regional_points[0]), 
                                     torch.linspace(y_min, y_max, regional_points[1]), 
                                     indexing='ij')
            regional_points = torch.stack(meshgrid, dim=-1).reshape(-1,2).to(physical_points.device)

            if periodic:
                _domain_shifts = domain_shifts((2.0,2.0)).to(physical_points.device)
                regional_points = rescale(regional_points, (-1,1))
            else:
                _domain_shifts = None
                regional_points = rescale(regional_points, (-1,1))
        if fix_radius:
            radii = torch.full([regional_points.shape[0]], fix_radius).to(regional_points.device)
        else:
            radii = minimal_support(regional_points, _domain_shifts) #[num_regional_points]
        # compute physical to regional edges
        p2r_edges = radius_bipartite_graph(
            physical_points, 
            regional_points, 
            radii * overlap_factor_p2r, 
            periodic
        )
        physical_to_regional = Graph.bipartite_with_pos_encode(
            p2r_edges, 
            src_pos   = physical_points, 
            dst_pos   = regional_points,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            dst_ndata       = (radii * overlap_factor_p2r)[:,None],
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        r2r_edges, domains = hierarchical_graph(
            regional_points, 
            level=regional_level, 
            domain_shifts=_domain_shifts)

        regional_to_regional = Graph.with_pos_encode(
            edges = r2r_edges, 
            pos = regional_points,
            ndata = (overlap_factor_r2p * radii)[:,None],
            domain_shifts = _domain_shifts,
            domain_edges = domains,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        # compute regional to physical edges
        # Actually, the r2p_edges can directly use the r2p_edges = torch.flip(p2r_edges, dims=(0,))
        # Under this circumstance, we don't need to recaculate the raddi anymore, use the calculated results above
        if fix_radius:
            radii = torch.full([physical_points.shape[0]], fix_radius).to(physical_points.device)
        else:
            radii = minimal_support(physical_points, _domain_shifts) #[num_regional_points]
        if output_points is None:
            #r2p_edges = torch.flip(p2r_edges, dims=(0,))
            r2p_edges = radius_bipartite_graph(
                regional_points, 
                physical_points, 
                radii * overlap_factor_r2p, 
                periodic
            )
        else:
            r2p_edges = radius_bipartite_graph(
                output_physical_points, 
                regional_points, 
                radii * overlap_factor_r2p, 
                periodic
            )

        regional_to_physical = Graph.bipartite_with_pos_encode(
            r2p_edges, 
            src_pos   = regional_points, 
            dst_pos   = physical_points,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            dst_ndata       = (overlap_factor_r2p * radii)[:, None],
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        return cls(physical_to_regional, regional_to_regional, regional_to_physical)
    