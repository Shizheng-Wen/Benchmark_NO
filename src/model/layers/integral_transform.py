
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.segment_csr import segment_csr
from .mlp import LinearChannelMLP
from typing import Optional, Literal, Dict

############
# Integral Transform (GNO)
############

class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)
    
    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Supports CSR neighbor format and optional neighbor sampling during training.

    Parameters
    ----------
    channel_mlp : torch.nn.Module, optional
        Pre-initialized MLP for the kernel k.
    channel_mlp_layers : list, optional
        Layer sizes for the kernel MLP if channel_mlp is not provided.
    channel_mlp_non_linearity : callable, default F.gelu
        Non-linearity for the kernel MLP.
    transform_type : str, default 'linear'
        Type of integral transform ('linear', 'nonlinear', etc.).
    use_attn : bool, default False
        Whether to use the attention mechanism.
    coord_dim : int, optional
        Coordinate dimension, required if use_attn is True.
    attention_type : Literal['cosine', 'dot_product'], default 'cosine'
        Type of attention mechanism.
    use_torch_scatter : bool, default True
        Whether to use torch_scatter backend for segment_csr if available.
    sampling_strategy : Optional[Literal['max_neighbors', 'ratio']], default None
        The neighbor sampling strategy to apply during training.
    max_neighbors : int, optional
        Maximum number of neighbors per node, required if strategy is 'max_neighbors'.
    sample_ratio : float, optional
        Ratio of neighbors to keep, required if strategy is 'ratio'.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_attn=None,
        attention_type='cosine',
        coord_dim=None,
        use_torch_scatter=True,
        # --- Neighbor sampling ---
        sampling_strategy: Optional[Literal['max_neighbors', 'ratio']] = None,
        max_neighbors: Optional[int] = None,
        sample_ratio: Optional[float] = None,
    ):
        super().__init__()

        # --- Store configuration ---
        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter
        self.use_attn = use_attn
        self.attention_type = attention_type
        self.sampling_strategy = sampling_strategy
        self.max_neighbors = max_neighbors
        self.sample_ratio = sample_ratio

        # --- Validate parameters ---
        if channel_mlp is None and channel_mlp_layers is None:
             raise ValueError("Either channel_mlp or channel_mlp_layers must be provided.")
        if self.transform_type not in ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"]:
            raise ValueError(f"Invalid transform_type: {transform_type}")
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")
            self.coord_dim = coord_dim # Store coord_dim only if use_attn is True
            if self.attention_type not in ['cosine', 'dot_product']:
                 raise ValueError(f"Invalid attention_type: {self.attention_type}")
        ## Validate sampling parameters
        if self.sampling_strategy == 'ratio':
            if self.sample_ratio is None:
                 raise ValueError("sample_ratio must be provided for 'ratio' sampling strategy.")
            if not (0.0 < self.sample_ratio <= 1.0):
                 raise ValueError(f"sample_ratio must be in (0.0, 1.0], got {self.sample_ratio}")
        elif self.sampling_strategy == 'max_neighbors':
            if self.max_neighbors is None:
                 raise ValueError("max_neighbors must be provided for 'max_neighbors' sampling strategy.")
            if self.max_neighbors <= 0:
                 raise ValueError(f"max_neighbors must be a positive integer, got {self.max_neighbors}")
        elif self.sampling_strategy is not None:
             raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}. Must be 'ratio', 'max_neighbors', or None.")

        # --- Initialize Modules ---
        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp

        ## Initialize attention projection if needed

        if self.use_attn and self.attention_type == 'dot_product':
            attention_dim = 64 
            self.query_proj = nn.Linear(self.coord_dim, attention_dim)
            self.key_proj = nn.Linear(self.coord_dim, attention_dim)
            self.scaling_factor = 1.0 / (attention_dim ** 0.5)

    def _apply_neighbor_sampling_csr(self, neighbors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies neighbor sampling based on the configured strategy (CSR format).
        """
        neighbors_index = neighbors["neighbors_index"]
        neighbors_row_splits = neighbors["neighbors_row_splits"]
        device = neighbors_index.device

        if neighbors_index.numel() == 0: return neighbors # No edges
        num_target_nodes = neighbors_row_splits.shape[0] - 1
        num_total_original_edges = neighbors_index.shape[0]

        # --- Strategy 1: Global Ratio Sampling ---
        if self.sampling_strategy == 'ratio':
            if self.sample_ratio is None or self.sample_ratio >= 1.0: return neighbors
            # (Ratio sampling logic remains the same)
            num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
            if (num_reps < 0).any() or torch.sum(num_reps) != num_total_original_edges:
                 raise ValueError("Invalid CSR structure before ratio sampling.")
            target_node_indices = torch.arange(num_target_nodes, device=device).repeat_interleave(num_reps)
            keep_mask = torch.rand(num_total_original_edges, device=device) < self.sample_ratio
            sampled_neighbors_index = neighbors_index[keep_mask]
            sampled_target_node_indices = target_node_indices[keep_mask]
            new_num_reps = torch.bincount(sampled_target_node_indices, minlength=num_target_nodes)
            sampled_neighbors_row_splits = torch.zeros(num_target_nodes + 1, dtype=neighbors_row_splits.dtype, device=device)
            torch.cumsum(new_num_reps, dim=0, out=sampled_neighbors_row_splits[1:])
        
        elif self.sampling_strategy == 'max_neighbors':
            if self.max_neighbors is None: return neighbors
            num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
            if (num_reps < 0).any() or torch.sum(num_reps) != num_total_original_edges:
                 raise ValueError("Invalid CSR structure before max_neighbors sampling.")

            needs_sampling_mask = num_reps > self.max_neighbors
            if not torch.any(needs_sampling_mask):
                return neighbors # No node exceeds the limit, return original
            
            keep_mask = torch.ones(num_total_original_edges, dtype=torch.bool, device=device)
            ## Iterate ONLY over nodes that need sampling
            nodes_to_sample_idx = torch.where(needs_sampling_mask)[0]
            for i in nodes_to_sample_idx:
                start = neighbors_row_splits[i]
                end = neighbors_row_splits[i+1]
                num_node_neighbors = int(num_reps[i]) # Original count for this node
                perm = torch.randperm(num_node_neighbors, device=device)
                # Select the local indices of neighbors to *keep* (the first max_neighbors)
                keep_local_indices = perm[:self.max_neighbors]
                local_keep_mask = torch.zeros(num_node_neighbors, dtype=torch.bool, device=device)
                # Mark the positions corresponding to kept local indices as True
                local_keep_mask[keep_local_indices] = True
                keep_mask[start:end] = local_keep_mask
            sampled_neighbors_index = neighbors_index[keep_mask]
            max_n_tensor = torch.full_like(num_reps, self.max_neighbors) # Tensor of max_neighbors
            new_num_reps = torch.minimum(num_reps, max_n_tensor)
            sampled_neighbors_row_splits = torch.zeros(num_target_nodes + 1, dtype=neighbors_row_splits.dtype, device=device)
            torch.cumsum(new_num_reps, dim=0, out=sampled_neighbors_row_splits[1:])
        else:
            return neighbors # No sampling needed
        
        return {
            "neighbors_index": sampled_neighbors_index,
            "neighbors_row_splits": sampled_neighbors_row_splits
        }

    def _segment_softmax(self, attention_scores, splits):
        """
        apply soft_max on every regional node neighbors.

        Parameters：
        - attention_scores: [num_neighbors]，attention scores
        - splits: neighbors split information

        Return：
        - attention_weights: [num_neighbors]，normalized attention scores
        """
        max_values = segment_csr(
            attention_scores, splits, reduce='max', use_scatter=self.use_torch_scatter
        )
        max_values_expanded = max_values.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_scores = attention_scores - max_values_expanded
        exp_scores = torch.exp(attention_scores)
        sum_exp = segment_csr(
            exp_scores, splits, reduce='sum', use_scatter=self.use_torch_scatter
        )
        sum_exp_expanded = sum_exp.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_weights = exp_scores / sum_exp_expanded
        return attention_weights

    def forward(self, 
                y: torch.Tensor, 
                neighbors: Dict[str, torch.Tensor], 
                x: Optional[torch.Tensor] = None, 
                f_y: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y

        ## --- Apply Neighbor Sampling ---
        current_neighbors = neighbors
        if self.training and self.sampling_strategy is not None:
            current_neighbors = self._apply_neighbor_sampling_csr(neighbors)

        
        neighbors_index = current_neighbors["neighbors_index"]
        neighbors_row_splits = current_neighbors["neighbors_row_splits"]
        num_query_nodes = neighbors_row_splits.shape[0] - 1

        # --- Gather features ---
        rep_features = y[neighbors_index]

        # --- Batching ---
        ## batching only matters if f_y (latent embedding) values are provided
        batched = False
        in_features = None
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors_index, :]
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors_index]
            else:
                raise ValueError(f"f_y has unexpected ndim: {f_y.ndim}")
        
        # --- Prepare 'self' features ---
        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        # --- Attention Logic ---
        attention_weights = None
        if self.use_attn:
            query_coords = self_features[:, :self.coord_dim]
            key_coords = rep_features[:, :self.coord_dim]
            if self.attention_type == 'dot_product':
                query = self.query_proj(query_coords)  # [num_neighbors, attention_dim]
                key = self.key_proj(key_coords)        # [num_neighbors, attention_dim]
                attention_scores = torch.sum(query * key, dim=-1) * self.scaling_factor  # [num_neighbors] 
            elif self.attention_type == 'cosine':
                query_norm = F.normalize(query_coords, p=2, dim=-1)
                key_norm = F.normalize(key_coords, p=2, dim=-1)
                attention_scores = torch.sum(query_norm * key_norm, dim=-1)  # [num_neighbors]
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")
            attention_weights = self._segment_softmax(attention_scores, neighbors_row_splits)
        else:
            attention_weights = None
        
        # --- Prepare input for the kernel MLP ---
        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        # --- Apply Kernel MLP ---
        rep_features = self.channel_mlp(agg_features) # TODO:这一步累计的计算图巨大[280468]

        # --- Apply f_y multiplication ---
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features
        
        # --- Apply attention weights ---
        if self.use_attn:
            rep_features = rep_features * attention_weights.unsqueeze(-1)
        
        # --- Apply Integration Weights ---
        if weights is not None:
            assert weights.ndim == 1, "Weights must be of dimension 1 in all cases"
            nbr_weights = weights[neighbors_index]
            # repeat weights along batch dim if batched
            if batched:
                nbr_weights = nbr_weights.repeat(
                    [batch_size] + [1] * nbr_weights.ndim
                )
            rep_features = nbr_weights * rep_features
            reduction = "sum"
        else:
            reduction = "mean" if not self.use_attn else "sum"

        # --- Aggregate using segment_csr ---

        splits = neighbors_row_splits
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(rep_features, splits, reduce=reduction, use_scatter=self.use_torch_scatter)

        return out_features

