import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from torch_geometric.data import Data

from .gno_utils.activation import Activation

from typing import Literal
import importlib

class MLP(nn.Module):
    """MLP with configurable number of layers and activation function

        Parameters:
        -----------
            x: torch.FloatTensor
                node features [num_nodes, num_features]
        Returns:
        --------
            y: torch.FloatTensor
                node labels [num_nodes, num_classes]
    """
    def __init__(self, num_features, num_classes,
        num_hidden=64, num_layers=3, activation="relu", input_dropout=0., dropout=0., bn=False, res=False):
        super().__init__()
        self.layers     = nn.ModuleList([nn.Linear(num_features, num_hidden)])
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_classes))
        self.activation = Activation(activation)
        self.input_dropout    = nn.Dropout(input_dropout) if input_dropout > 0 else Identity()
        self.dropout          = nn.Dropout(dropout) if dropout > 0 else Identity()
        if bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(num_features)])
            for i in range(num_layers-2):
                self.bns.append(nn.BatchNorm1d(num_hidden))
            self.bns.append(nn.BatchNorm1d(num_hidden))
        else:
            self.bns = None
        if res:
            self.linear = nn.Linear(num_features, num_classes)
        else:
            self.linear = None
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
    def forward(self, x):
        input = x
        x = self.input_dropout(x)
        x = self.bns[0](x) if self.bns is not None else x
        for i,layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.bns[i](x) if self.bns is not None else x
        x = self.layers[-1](x)
        if self.linear is not None:
            x = x + self.linear(input)
        return x

## Implementation from Neuraloperator: https://github.com/neuraloperator/neuraloperator
class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x

# Reimplementation of the ChannelMLP class using Linear instead of Conv
class LinearChannelMLP(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x

class FourierMLP(nn.Module):
    """Fourier encoder for nodes and edges that returns the latent features"""
    def __init__(self, node_num_input_features: int, 
                edge_num_input_features: int,
                L: int = 4,
                num_latent_features: int=64, 
                num_hidden: int = 64,
                num_layers: int = 3, 
                activation: str = "relu", 
                dropout: float = 0.0, 
                bn: bool = False, 
                res: bool = False):
        super().__init__()
        self.L = L
        self.node_num_input_features = node_num_input_features
        self.edge_num_input_features = edge_num_input_features
        self.num_latent_features = num_latent_features
        self.num_layers = num_layers
        self.activation = Activation(activation) 
        self.mlp_node = MLP(node_num_input_features * (4 * L - 1), 
                           num_latent_features, 
                           num_hidden=num_hidden, 
                           num_layers=num_layers, 
                           activation=activation, 
                           dropout=dropout, 
                           bn=bn, 
                           res=res)
        self.mlp_edge = MLP(edge_num_input_features * (4 * L - 1), 
                           num_latent_features, 
                           num_hidden=num_hidden, 
                           num_layers=num_layers, 
                           activation=activation, 
                           dropout=dropout, 
                           bn=bn, 
                           res=res)

    def forward(self, graph: Data):
        Nfeatures = [graph.x]
        Efeatures = [graph.edge_attr]
        for i in range(self.L, 0, -1):
            omega = 1/i
            Nfeatures.append(torch.cos(omega * graph.x))
            Efeatures.append(torch.cos(omega * graph.edge_attr))
            Nfeatures.append(torch.sin(omega * graph.x))
            Efeatures.append(torch.sin(omega * graph.edge_attr))
        Nfeatures = torch.cat(Nfeatures, dim=-1)
        Efeatures = torch.cat(Efeatures, dim=-1)
        return self.mlp_node(Nfeatures), self.mlp_edge(Efeatures)
