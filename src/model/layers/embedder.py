import torch 
import torch.nn as nn
from typing import Union, Tuple, Optional
from dataclasses import dataclass, asdict, replace
from .mlp import AugmentedMLP, AugmentedMLPConfig
from .message_passing import MessagePassingLayerConfig
from src.utils.dataclass import shallow_asdict
from src.utils.pair import make_pair, is_pair

class FeatureEncoderLayer(nn.Module):
    edge_fn:Optional[AugmentedMLP]
    node_fn:Optional[AugmentedMLP]
    def __init__(self,
                 node_input_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_input_size:Optional[int] = None,
                 node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_output_size:Optional[int] = None,
                 use_node_fn:bool = True, 
                 use_edge_fn:bool = True,
                 edge_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                 node_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                 ):
        super().__init__()

        if use_edge_fn:
            self.edge_fn = AugmentedMLP.from_config(
                input_size = edge_input_size,
                output_size = edge_output_size,
                config = edge_fn_config
            ) 
            
        else:
            self.edge_fn = None 
        if use_node_fn:
            if is_pair(node_input_size):
                self.node_fn = nn.ModuleList([
                    AugmentedMLP.from_config(
                        input_size = node_input_size[0],
                        output_size = make_pair(node_output_size)[0],
                        config = node_fn_config
                    ),
                    AugmentedMLP.from_config(
                        input_size = node_input_size[1],
                        output_size = make_pair(node_output_size)[1],
                        config = node_fn_config
                    )
                ])
            else:
                assert not is_pair(node_output_size), f"node_output_size:{node_output_size} should not be pair"
                self.node_fn = AugmentedMLP.from_config(
                    input_size = node_input_size,
                    output_size = node_output_size,
                    config = node_fn_config
                )
        else:
            self.node_fn = None

    def forward(self, ndata, edata, **kwargs
                )->Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Parameters
        ----------
        ndata:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            if is_bipartite:
                (src_ndata, dst_ndata)
                src_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_src_nodes, node_input_size]
                dst_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_dst_nodes, node_input_size]
            else:
                ndata:torch.Tensor
                    torch.Tensor of shape [..., n_nodes, node_input_size]
            Input node features
        edata:torch.Tensor
            Input edge features
            torch.Tensor of shape [..., n_edges, edge_input_size]
        kwargs:dict
            Additional arguments
            
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Output node features
            if is_bipartite:
                (src_ndata, dst_ndata)
                src_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_src_nodes, node_output_size]
                dst_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_dst_nodes, node_output_size]
            else:
                ndata:torch.Tensor
                    torch.Tensor of shape [..., n_nodes, node_output_size]
        torch.Tensor
            Output edge features
        """
        if self.edge_fn is not None:
            edata = self.edge_fn(edata, **kwargs)
        if self.node_fn is not None:
            if isinstance(self.node_fn, nn.ModuleList):
                assert is_pair(ndata), f"ndata should be a pair, got {ndata}"
                # is_bipartite
                src_ndata, dst_ndata = ndata
                
                src_ndata = self.node_fn[0](src_ndata, **kwargs)
                dst_ndata = self.node_fn[1](dst_ndata, **kwargs)
    
                ndata = (src_ndata, dst_ndata)
            else:
                assert not is_pair(ndata), f"ndata should not be a pair, got {ndata}"
                # not is_bipartite
                ndata = self.node_fn(ndata, **kwargs)
        
        return ndata, edata

    @classmethod
    def from_config(cls,
                      node_input_size:Optional[Union[int, Tuple[int, int]]],
                      edge_input_size:int,
                      node_output_size:Optional[Union[int, Tuple[int, int]]],
                      edge_output_size:int,
                      config:MessagePassingLayerConfig):
    
        kwargs = shallow_asdict(config)
        kwargs.pop("aggregate")
        kwargs.pop("aggregate_normalization")
        return cls(node_input_size, 
                    edge_input_size, 
                    node_output_size, 
                    edge_output_size, 
                    **kwargs)
    