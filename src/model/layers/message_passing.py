import torch 
import torch.nn as nn
from typing import Union, Tuple, Optional, Any
from dataclasses import dataclass, asdict, replace, is_dataclass, field
from .mlp import AugmentedMLP, AugmentedMLPConfig
from src.utils.dataclass import shallow_asdict
from src.graph import Graph
from src.utils.pair import make_pair, is_pair

############
# Config
############

@dataclass
class MessagePassingLayerConfig:
    edge_fn_config:AugmentedMLPConfig = field(default_factory=AugmentedMLPConfig)
    node_fn_config:AugmentedMLPConfig = field(default_factory=AugmentedMLPConfig)
    aggregate:str = "mean"
    aggregate_normalization:Optional[float] = None
    use_node_fn:bool = True
    use_edge_fn:bool = True

############
# Message Passing Layer
############

class MessagePassingLayer(nn.Module):
    edge_fn:AugmentedMLP
    node_fn:Union[AugmentedMLP, nn.ModuleDict]
    aggregate:str
    aggregate_normalization:Optional[float]

    node_input_size:int 
    edge_input_size:int
    def __init__(self,
                 node_input_size:Optional[Union[int,Tuple[int,int]]] = None, 
                 edge_input_size:Optional[int] = None,
                 node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_output_size:Optional[int] = None,
                 aggregate:str = "mean",
                 aggregate_normalization:Optional[float] = None,
                 use_node_fn:bool = True, 
                 use_edge_fn:bool = True,
                 edge_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                 node_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                ):
        """
        Parameters
        ----------
        node_input_size:Optional[Union[int,Tuple[int,int]]]
            Input size of node features
            if None, there is no node update 
            elif Tuple[int,int], it is bipartite
            elif int, it is homogenous
        edge_input_size:int
            Input size of edge features
        node_output_size:Optional[Union[int,Tuple[int,int]]]
            Output size of node features
            if None, there is no node update
            elif Tuple[int,int], it is bipartite
            elif int, it is homogenous
        edge_output_size:int
            Output size of edge features
        aggregate:str
            Aggregation function
        aggregate_normalization:Optional[float]
            Normalization factor for aggregation
        edge_fn_config:AugmentedMLPConfig
            Configuration for edge function
        node_fn_config:AugmentedMLPConfig
            Configuration for node function
        
        """
        
        super().__init__()


        
        if use_edge_fn:
            assert edge_output_size is not None, "edge_output_size should be provided"
            # assert node_input_size is not None, "node_input_size should be provided"
            if edge_input_size is None:
                edge_input_size = 0 
            
            self.edge_fn = AugmentedMLP.from_config(
                input_size = edge_input_size + sum(make_pair(node_input_size)),
                output_size = edge_output_size,
                config = edge_fn_config
            )
        else:
            self.edge_fn = None

        if use_node_fn:
            if is_pair(node_input_size):
                for i in range(2):
                    if node_input_size[i] is None:
                        node_input_size[i] = 0
                self.node_fn = nn.ModuleList([
                    AugmentedMLP.from_config(
                        input_size  = edge_output_size + node_input_size[0],
                        output_size = make_pair(node_output_size)[0],
                        config      = node_fn_config
                    ),
                    AugmentedMLP.from_config(
                        input_size  = edge_output_size + node_input_size[1],
                        output_size = make_pair(node_output_size)[1],
                        config      = node_fn_config
                    )
                ])
            else:
                if node_input_size is None:
                    node_input_size = 0
                self.node_fn = AugmentedMLP.from_config(
                    input_size  = edge_output_size * 2 + node_input_size,
                    output_size = node_output_size,
                    config      = node_fn_config
                )
            
        else:
            self.node_fn = None
    
        self.aggregate = aggregate
        self.aggregate_normalization = aggregate_normalization
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

    @property 
    def is_bipartite(self):
        return isinstance(self.node_input_size, (list, tuple))

    def forward(self, 
                graph:Graph, 
                ndata:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                edata:Optional[torch.Tensor] = None,
                **kwargs
                )->Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Parameters
        ----------
        graph:Graph
            homogenous or bipartite graph
        ndata:torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            ND tensor of shape [..., n_nodes, node_input_size]
        edata:torch.Tensor
            ND tensor of shape [..., n_edges, edge_input_size]

        Returns
        -------
        if is_bipartite:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            ((src_ndata, dst_ndata), edata)
            src_ndata: [..., n_src_nodes, node_output_size]
            dst_ndata: [..., n_dst_nodes, node_output_size]
            edata: [..., n_edges, edge_output_size]
        else:
            Tuple[torch.Tensor, torch.Tensor]
            (ndata, edata)
            ndata: [..., n_nodes, node_output_size]
            edata: [..., n_edges, edge_output_size]
        """
        # assert graph.is_bipartite == self.is_bipartite, f"graph.is_bipartite({graph.is_bipartite}) != self.is_bipartite({self.is_bipartite})"
        assert make_pair(graph.num_nodes) == tuple(x.shape[-2] for x in make_pair(ndata)), \
            f"graph.num_nodes({make_pair(graph.num_nodes)}) != ndata.shape[-2]({tuple(x.shape[0] for x in make_pair(ndata))})"
        # assert make_pair(self.node_input_size) == tuple(x.shape[-1] for x in make_pair(ndata)), \
        #     f"self.node_input_size({self.node_input_size}) != ndata.shape[-2]({make_pair(self.node_input_size)})"
        if edata is not None:
            assert graph.num_edges      == edata.shape[-2], f"graph.num_edges({graph.num_edges}) != edata.shape[-2]({edata.shape})"
            assert self.edge_input_size == edata.shape[-1], f"self.edge_input_size({self.edge_input_size}) != edata.shape[-1]({edata.shape[1]})"
        
        if self.edge_fn is not None:
            def message_fn(src_ndata, dst_ndata, edata):
                x = []
                if src_ndata is not None:
                    x.append(src_ndata)
                if dst_ndata is not None:
                    x.append(dst_ndata)
                if edata is not None:
                    x.append(edata)
                x = torch.cat(x, -1)
                return self.edge_fn(x, **kwargs)
        
            edata = graph.message(message_fn, ndata=ndata, edata=edata)
        
        if self.node_fn is not None:

            if graph.is_bipartite:

                def src_aggregate_fn(edata, ndata):
                    x = [edata]
                    if ndata is not None:
                        x.append(ndata)
                    x = torch.cat(x, -1)
                    return self.node_fn[0](x, **kwargs)

                def dst_aggregate_fn(edata, ndata):
                    x = [edata]
                    if ndata is not None:
                        x.append(ndata)
                    x = torch.cat(x, -1)
                    return self.node_fn[1](x, **kwargs)

                src_ndata, dst_ndata = graph.aggregate((src_aggregate_fn, dst_aggregate_fn), 
                                                      edata=edata, 
                                                      ndata=ndata,
                                                      reduce=self.aggregate)
            
                if self.aggregate_normalization is not None:
                    src_ndata = src_ndata / self.aggregate_normalization
                    dst_ndata = dst_ndata / self.aggregate_normalization

                ndata = (src_ndata, dst_ndata)

            else:

                def aggregate_fn(src_edata, dst_edata, ndata):
                    x = torch.cat([src_edata, dst_edata, ndata], -1)
                    return self.node_fn(x, **kwargs)
                
                ndata = graph.aggregate(aggregate_fn, edata=edata, ndata=ndata, reduce=self.aggregate)

                if self.aggregate_normalization is not None:
                    ndata = ndata / self.aggregate_normalization

        else:

            if graph.is_bipartite:
                src_ndata, dst_ndata = graph.aggregate((lambda e,n:e, lambda e,n:e), 
                                                      edata=edata, 
                                                      ndata=ndata,
                                                      reduce=self.aggregate)
                ndata = (src_ndata, dst_ndata)
            else:
                ndata = graph.aggregate(lambda e_u, e_v,n:torch.cat([e_u, e_v], -1), edata=edata, ndata=ndata, reduce=self.aggregate)

        return ndata, edata

    @classmethod 
    def from_config(cls, 
                    node_input_size:Optional[Union[int, Tuple[int,int]]] = None,
                    edge_input_size:Optional[int] = None,
                    node_output_size:Optional[Union[int, Tuple[int,int]]] = None,
                    edge_output_size:Optional[int] = None,
                    config:MessagePassingLayerConfig = MessagePassingLayerConfig()):

        return cls(node_input_size, 
                   edge_input_size, 
                   node_output_size, 
                   edge_output_size, 
                   **shallow_asdict(config))
