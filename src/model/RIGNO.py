import torch 
import torch.nn as nn
from typing import Tuple, Optional, Union
from .layers.deepgnn import DeepGraphNN, DeepGraphNNConfig
from src.utils.pair import make_pair
from src.utils.dataclass import safe_replace
from src.graph import RegionInteractionGraph, Graph

class Physical2Regional2Physical(nn.Module):
    """
    Regional Interaction Graph Neural Operator
    """
    def __init__(self,
                 input_size:int, 
                 output_size:int,
                 rigraph:RegionInteractionGraph,
                 variable_mesh:bool = False,
                 drop_edge:float = 0.0,
                 config:DeepGraphNNConfig = DeepGraphNNConfig()):
        super().__init__()
        self.drop_edge = drop_edge
        self.input_size = input_size 
        self.output_size = output_size
        self.node_latent_size = config.node_latent_size
        
        self.encoder   = self.init_encoder(input_size, rigraph, config)
        self.processor = self.init_processor(rigraph, config)
        self.decoder   = self.init_decoder(output_size, rigraph, variable_mesh, config)
        
    def init_encoder(self, node_input_size:int, 
                        rigraph:RegionInteractionGraph,
                        config):
        return DeepGraphNN.from_config(
            node_input_size = (node_input_size + rigraph.physical_to_regional.ndim[0], rigraph.physical_to_regional.ndim[1]),
            edge_input_size = rigraph.physical_to_regional.edim,
            config = safe_replace(config,
                    use_node_encode = True, # it will be broadcasted to pair
                    use_edge_encode = True,
                    use_node_decode = False, 
                    use_edge_decode = False,
                    num_message_passing_steps = 1)
            )   

    def init_processor(self, 
                       rigraph:RegionInteractionGraph, 
                       config):
        return DeepGraphNN.from_config(
            node_input_size = self.node_latent_size,
            edge_input_size = rigraph.regional_to_regional.edim,
            config = safe_replace(config,
                             use_node_encode=True, 
                             use_edge_encode=True,
                             use_node_decode=False,
                             use_edge_decode=False    )
        )

    def init_decoder(self, output_size:int, 
                     rigraph:RegionInteractionGraph,
                     variable_mesh:bool, 
                     config):
        return DeepGraphNN.from_config(
            node_input_size  = (self.node_latent_size, self.node_latent_size),
            edge_input_size  = rigraph.regional_to_physical.edim,
            node_output_size = output_size,
            config = safe_replace(config,
                      use_node_encode = (False, False) if variable_mesh else (True, False),
                      use_edge_encode = True,
                      use_node_decode = (True, False),
                      use_edge_decode = False,  
                      num_message_passing_steps = 1,
                      )
        )

    def encode(self, graph:Graph, 
                pndata:Optional[torch.Tensor] = None, 
                condition:Optional[float] = None
                )->torch.Tensor:
        """
        Parameters
        ----------
        graph:Graph
            physical to regional graph, a bipartite graph
        pndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_physical_nodes, node_input_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        if pndata is None:
            assert self.input_size == 0, f"pndata is None, but self.input_size({self.input_size}) != 0"
        else:
            assert pndata.shape[-1] == self.input_size, f"pndata.shape[-1]({pndata.shape[-1]}) != self.input_size({self.input_size})"
        if self.training:
            drop_edge = self.drop_edge
        else:
            drop_edge = 0.0
        graph       = graph.drop_edge(drop_edge)
        graph.batch_broadcast(batch_size=pndata.shape[0], ndata_dim=True, edata_dim=True)
        pndata      = torch.cat([graph.get_ndata()[0], pndata], -1) if pndata is not None else graph.get_ndata()[0] 
        rndata      = graph.get_ndata()[1] 
        ndata       = (pndata, rndata)
        edata       = graph.get_edata()
        ndata, _ = self.encoder(graph,
                                    ndata,
                                    edata,
                                    condition = condition)
        pndata, rndata = ndata

        return pndata, rndata

    def process(self, graph:Graph, 
                rndata:torch.Tensor, 
                condition:Optional[float] = None
                )->torch.Tensor:
        """
        Parameters
        ----------
        graph:Graph
            regional to regional graph, a homogeneous graph
        rndata:torch.Tensor
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        assert rndata is not None, "rndata should not be None"
        assert rndata.shape[-1] == self.node_latent_size, f"rndata.shape[-1]({rndata.shape[-1]}) != self.node_latent_size({self.node_latent_size})"
        if self.training:
            drop_edge = self.drop_edge
        else:
            drop_edge = 0.0
        graph       = graph.drop_edge(drop_edge)
        graph.batch_broadcast(batch_size=rndata.shape[0], ndata_dim=False, edata_dim=True)
        edata       = graph.get_edata()

        rndata, _ = self.processor(graph, rndata, edata, condition = condition)

        return rndata

    def decode(self, graph:Graph, 
                rndata:torch.Tensor, 
                pndata:torch.Tensor,
                condition:Optional[float] = None
                )->torch.Tensor:
        """
        Parameters
        ----------
        graph:Graph
            regional to physical graph, a bipartite graph
        rndata:torch.Tensor
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The physical node data of shape [..., n_physical_nodes, output_size]
        """
        assert rndata is not None, "rndata should not be None"
        assert rndata.shape[-1] == self.node_latent_size, f"rndata.shape[-1]({rndata.shape[-1]}) != self.node_latent_size({self.node_latent_size})"
        
        if self.training:
            drop_edge = self.drop_edge
        else:
            drop_edge = 0.0
        graph       = graph.drop_edge(drop_edge)
        ndata = (rndata, pndata)
        graph.batch_broadcast(batch_size=pndata.shape[0], ndata_dim=True, edata_dim=True)
        edata       = graph.get_edata()
      
        ndata, _ = self.decoder(graph, ndata, edata, condition = condition)

        rndata, pndata = ndata

        assert pndata.shape[-1] == self.output_size, f"pndata.shape[-1]({pndata.shape[-1]}) != self.output_size({self.output_size})"

        return pndata

    def forward(self, 
                graphs:RegionInteractionGraph,
                pndata:Optional[torch.Tensor] = None, 
                condition:Optional[float] = None
                )->torch.Tensor:
        """
        Parameters
        ----------
        graphs:RegionInteractionGraph
        pndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_physical_nodes, node_input_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The output tensor of shape [..., n_physical_nodes, output_size]
        """
        
        pndata, rndata = self.encode(graphs.physical_to_regional, pndata, condition)

        rndata = self.process(graphs.regional_to_regional, rndata, condition)

        pndata = self.decode(graphs.regional_to_physical, rndata, pndata, condition)

        return pndata
