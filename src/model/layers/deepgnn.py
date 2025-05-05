import torch 
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional
from .message_passing import MessagePassingLayer, MessagePassingLayerConfig
from .embedder import FeatureEncoderLayer
from src.utils.pair import make_pair, is_pair
from src.utils.dataclass import shallow_asdict, safe_replace
from src.graph import Graph

############
# Config
############

@dataclass
class DeepGraphNNConfig:
    node_latent_size:int = 16 #Union[int, Tuple[int,int]] = 16 # 64
    edge_latent_size:int = 16 # 64
    num_message_passing_steps:int = 3
    use_node_encode:bool = True
    use_edge_encode:bool = True
    use_node_decode:bool = True 
    use_edge_decode:bool = True
    mpconfig:MessagePassingLayerConfig = field(default_factory=MessagePassingLayerConfig)

############
# Deep Graph Neural Network
############

    
class DeepGraphNN(nn.Module):
    
    encoder:Optional[FeatureEncoderLayer]
    processor:nn.ModuleList # nn.ModuleList[MessagePassingLayer]
    decoder:Optional[FeatureEncoderLayer]

    node_input_size:int 
    edge_input_size:int
    node_output_size:int
    edge_output_size:int
    def __init__(self, 
                 node_input_size:Union[int,Tuple[int,int]], 
                 node_latent_size:Union[int,Tuple[int,int]],
                 edge_input_size:Optional[int] = None,  
                 edge_latent_size:Optional[int] = None,
                 node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_output_size:Optional[int] = None,
                 num_message_passing_steps:int = 3,
                 use_node_encode:Union[bool,Tuple[bool,bool]] = True,
                 use_edge_encode:bool = True,
                 use_node_decode:Union[bool,Tuple[bool,bool]] = True, 
                 use_edge_decode:bool = True,
                 mpconfig:MessagePassingLayerConfig = MessagePassingLayerConfig()):
        """
        Parameters
        ----------
        node_input_size:Union[int,Tuple[int,int]]
            When it's bipartite, it should be a pair of int
        edge_input_size:int
            The size of edge features
        node_latent_size:Union[int,Tuple[int,int]]
            When it's bipartite, it's not required to be a pair, it can broadcast automatically
        edge_latent_size:int
            The latent size of edge features
        node_output_size:Optional[Union[int,Tuple[int,int]]]
            When it's bipartite, it's not required to be a pair, it can broadcast automatically
            if use_node_decode if False, it's not required to be provided, it will be automatically set to node_latent_size
        edge_output_size:Optional[int]
            When it's bipartite, it's not required to be a pair, it can broadcast automatically
            if use_edge_decode if False, it's not required to be provided, it will be automatically set to edge_latent_size
        num_message_passing_steps:int
            The number of message passing steps, default to 3
        use_node_encode:Union[bool,Tuple[bool,bool]]
            Whether to use node encoder
            if bipartite, it can be a pair of bool, it can be broadcasted automatically
            else, it should be a single bool
        use_edge_encode:bool
            Whether to use edge encoder
        use_node_decode:Union[bool,Tuple[bool,bool]]
            Whether to use node decoder
            if bipartite, it can be a pair of bool, it can be broadcasted automatically
            else, it should be a single bool
        use_edge_decode:bool
            Whether to use edge decoder
        mpconfig:MessagePassingLayerConfig
            The config of message passing layers

        """
        super().__init__()
        assert num_message_passing_steps > 0, f"num_message_passing_steps should be greater than 0, got {num_message_passing_steps}"

        is_bipartite = is_pair(node_input_size)

        # FeatureEncoderLayerCls = FeatureEncoderLayer if not is_bipartite else FeatureEncoderBipartiteLayer
        # MessagePassingLayerCls = MessagePassingLayer if not is_bipartite else MessagePassingBipartiteLayer
        broadcast_fn = make_pair if is_bipartite else lambda x:x
        
        if any(make_pair(use_node_encode)) or any(make_pair(use_edge_encode)):
            # any use_node_encode or use_edge_encode is True
            self.encoder = FeatureEncoderLayer.from_config(
                node_input_size = node_input_size,  
                edge_input_size = edge_input_size,
                node_output_size = broadcast_fn(node_latent_size),
                edge_output_size = edge_latent_size,
                config = safe_replace(mpconfig,
                                    node_fn_config = safe_replace(mpconfig.node_fn_config,
                                                                use_conditional_norm = False),
                                    edge_fn_config = safe_replace(mpconfig.edge_fn_config,
                                                                use_conditional_norm = False),
                                    use_node_fn = use_node_encode,
                                    use_edge_fn = use_edge_encode
                                    )
            )
            
        else:
            self.encoder  = None
        
        node_input_size = node_latent_size if use_node_encode else node_input_size
        edge_input_size = edge_latent_size if use_edge_encode else edge_input_size

        self.processor = nn.ModuleList([
            MessagePassingLayer.from_config(
                node_input_size = broadcast_fn(node_input_size),
                edge_input_size = edge_input_size,
                node_output_size = broadcast_fn(node_latent_size),
                edge_output_size = edge_latent_size,    
                config = mpconfig
            )  
        ])
        for _ in range(num_message_passing_steps-1):
            self.processor.append(
                MessagePassingLayer.from_config(
                    node_input_size = broadcast_fn(node_latent_size),
                    edge_input_size = edge_latent_size,
                    node_output_size = broadcast_fn(node_latent_size),
                    edge_output_size = edge_latent_size,
                    config = mpconfig
                )
            )
        
        if any(make_pair(use_node_decode)) or any(make_pair(use_edge_decode)):
            # any use_node_decode or use_edge_decode is True
            self.decoder = FeatureEncoderLayer.from_config(
                node_input_size = broadcast_fn(node_latent_size),
                edge_input_size = edge_latent_size,
                node_output_size = broadcast_fn(node_output_size),
                edge_output_size = edge_output_size,
                config = safe_replace(mpconfig,
                                node_fn_config = safe_replace(mpconfig.node_fn_config,
                                                        use_layer_norm = False,
                                                        use_conditional_norm = False),
                                edge_fn_config = safe_replace(mpconfig.edge_fn_config,
                                                        use_layer_norm = False,
                                                        use_conditional_norm = False),
                                use_node_fn = use_node_decode,
                                use_edge_fn = use_edge_decode,
                )
            )
        else:
            self.decoder = None

        if is_bipartite:
            self.node_output_size = tuple(lambda o,l,d:o if d else l for o,l,d in zip(
                                            make_pair(node_output_size), 
                                            make_pair(node_latent_size), 
                                            make_pair(use_node_decode)))
        else:
            self.node_output_size = node_output_size if use_node_decode else node_latent_size
        self.edge_output_size = edge_output_size if use_edge_decode else edge_latent_size
    
    def forward(self, 
                graph:Graph,
                ndata:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edata:Optional[torch.Tensor] = None,
                condition:Optional[float] = None
                )->Union[
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]  
                ]:
        if self.encoder is not None:
            ndata, edata = self.encoder(ndata, edata, condition=condition)
    
        for layer in self.processor:
            ndata, edata = layer(graph, ndata, edata, condition=condition)
        
        if self.decoder is not None:
            ndata, edata = self.decoder(ndata, edata, condition=None)

        return ndata, edata
    
    @classmethod 
    def from_config(cls,
                    node_input_size:Union[int,Tuple[int,int]], 
                    edge_input_size:int,  
                    node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                    edge_output_size:Optional[int] = None,
                    config:DeepGraphNNConfig = DeepGraphNNConfig()):
        return cls(node_input_size = node_input_size,
                    edge_input_size = edge_input_size,
                    node_output_size = node_output_size,
                    edge_output_size = edge_output_size,
                     **shallow_asdict(config))
  