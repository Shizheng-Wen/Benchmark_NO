import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass, field

from .mlp import ChannelMLP
from .utils.neighbor_search import NeighborSearch
from .geoembed import GeometricEmbedding
from .integral_transform import IntegralTransform



############
# MAGNO Encoder
############
class MAGNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        # --- Configuration ---
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edgs = gno_config.precompute_edges
        self.use_geoembed = gno_config.use_geoembed
        self.coord_dim = gno_config.gno_coord_dim
        # --- Modules ---
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.graph_cache = None 
        
        # Determine GNO input kernel dimension based on config
        in_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = self.coord_dim  
        if gno_config.node_embedding:
            in_kernel_in_dim = self.coord_dim * 4 * 2 * 2  # 32
            coord_dim = self.coord_dim * 4 * 2
        if gno_config.in_gno_transform_type == "nonlinear" or gno_config.in_gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += in_channels

        # Prepare GNO channel MLP layers
        in_gno_channel_mlp_hidden_layers = gno_config.in_gno_channel_mlp_hidden_layers.copy()
        in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        in_gno_channel_mlp_hidden_layers.append(out_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.in_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            attention_type=gno_config.attention_type,
            coord_dim=coord_dim,
            sampling_strategy=gno_config.sampling_strategy,
            max_neighbors = gno_config.max_neighbors,
            sample_ratio=gno_config.sample_ratio,
        )

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=1
        )

        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=out_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                n_layers=1
            )
        
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.coord_dim = coord_dim
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)
    
    def forward(self, 
                x_coord: torch.Tensor,
                pndata: torch.Tensor,
                latent_tokens_coord: torch.Tensor,
                encoder_nbrs: Optional[list[List[any]]] = None) -> torch.Tensor:
        """
        Forward pass for the MAGNO Encoder.

        Args:
            x_coord: Physical node coordinates. Shape: [num_nodes, coord_dim]
            pndata: Physical node features. Shape: [batch_size, num_nodes, in_channels]
            latent_tokens_coord: Coordinates of the target latent grid nodes. Shape: [num_latent_nodes, coord_dim]
            encoder_nbrs: Optional precomputed neighbor lists for each scale shared by all batch item.
                          Required if self.precompute_edges is True.
                          Format: List[neighbors_scale_0, neighbors_scale_1,...]
                          List length is num_scales.

        Returns:
            torch.Tensor: Encoded features on the latent grid. Shape: [batch_size, num_latent_nodes, out_channels]
        """
        ## --- Check dimensions ---
        if len(x_coord.shape) != 2 or x_coord.shape[1] != self.coord_dim:
            raise ValueError(f"Expected x_coord shape [num_nodes, {self.coord_dim}], "
                            f"got {x_coord.shape}")
        
        num_nodes = x_coord.shape[0]
        if len(pndata.shape) != 3 or pndata.shape[1] != num_nodes:
            raise ValueError(f"Expected pndata shape [batch_size, {num_nodes}, in_channels], "
                            f"got {pndata.shape}")
        
        if len(latent_tokens_coord.shape) != 2 or latent_tokens_coord.shape[1] != self.coord_dim:
            raise ValueError(f"Expected latent_tokens_coord shape [num_latent_nodes, {self.coord_dim}], "
                            f"got {latent_tokens_coord.shape}")
        
        if self.precompute_edgs and encoder_nbrs is None:
            raise ValueError("encoder_nbrs is required when precompute_edges is True")
        
        if encoder_nbrs is not None and len(encoder_nbrs) != len(self.scales):
            raise ValueError(f"Expected encoder_nbrs to have length {len(self.scales)} (number of scales), "
                            f"got {len(encoder_nbrs)}")
        # --- End of check dimensions ---
        
        batch_size = pndata.shape[0]
        device = pndata.device

        # --- 1. Precompute neighbor search  ---
        if self.precompute_edgs:
            if encoder_nbrs is None:
                raise ValueError("Precomputed neighbor lists must be provided when precompute_edges is True.")
            self.spatial_nbrs_scales = encoder_nbrs
        else:
            if self.graph_cache is None:
                self.spatial_nbrs_scales = []
                for scale in self.scales:
                    scaled_radius = self.gno_radius * scale
                    spatial_nbrs = self.nb_search(
                        data = x_coord,
                        queries = latent_tokens_coord,
                        radius = scaled_radius
                    )
                    self.spatial_nbrs_scales.append(spatial_nbrs)
                self.graph_cache = True 
        
        # --- 2. Lift input features ---
        pndata = pndata.permute(0, 2, 1)
        pndata = self.lifting(pndata).permute(0, 2, 1)  
        
        # --- 3. Encode features for each scale ---
        encoded_scales = []
        for idx, scale in enumerate(self.scales):
            spatial_nbrs = self.spatial_nbrs_scales[idx]
            encoded = self.gno(
                y = x_coord,
                x = latent_tokens_coord,
                f_y = pndata,
                neighbors=spatial_nbrs
            )

            ## Apply optional geometric embedding
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    input_geom = x_coord,
                    latent_queries = latent_tokens_coord,
                    spatial_nbrs = spatial_nbrs
                )
                
                geoembedding = geoembedding[None, :, :]
                geoembedding = geoembedding.repeat([batch_size, 1, 1])
                encoded = torch.cat([encoded, geoembedding], dim=-1)
                encoded = encoded.permute(0, 2, 1)
                encoded = self.recovery(encoded).permute(0, 2, 1)
            ## End of optional geometric embedding
            encoded_scales.append(encoded)

        # --- 4. Combine encoded features across scales ---
        if len(encoded_scales) == 1:
            encoded = encoded_scales[0]
        else:
            if self.use_scale_weights:
                scale_weights = scale_weights.permute(1, 0)                     # [num_scales, num_query_points]
                scale_weights = self.scale_weighting(latent_tokens_coord)       # [m, num_scales]
                scale_weights = self.scale_weight_activation(scale_weights)     # [m, num_scales]
                scale_weights = scale_weights.unsqueeze(0).unsqueeze(-1)        # [1, num_scales, num_query_points,1]
                scale_weights = scale_weights.repeat(batch_size, 1, 1, 1)       # [batch_size, num_scales, num_query_points,1]]
                encoded_scales_tensor = torch.stack(encoded_scales, dim=0)      # [num_scales, batch_size, num_query_points, feature_dim]
                encoded_scales_tensor = encoded_scales_tensor.permute(1, 0, 2, 3)
                weighted_encoded_scales = encoded_scales_tensor * scale_weights
                encoded = weighted_encoded_scales.sum(dim=1)                    # [batch_size, num_query_points, feature_dim]
            else:
                encoded = torch.stack(encoded_scales, dim=0).sum(dim=0)

        return encoded


############
# MAGNO Decoder
###########
class MAGNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        # --- Configuration ---
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edgs = gno_config.precompute_edges
        self.use_geoembed = gno_config.use_geoembed
        self.coord_dim = gno_config.gno_coord_dim
        # --- Modules ---
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.graph_cache = None 

         # Determine GNO input kernel dimension based on config
        out_kernel_in_dim = self.coord_dim * 2
        if gno_config.node_embedding:
            out_kernel_in_dim = self.coord_dim * 4 * 2 * 2  # 32
            self.coord_dim = self.coord_dim * 4 * 2
        if gno_config.out_gno_transform_type == "nonlinear" or gno_config.out_gno_transform_type == "nonlinear_kernelonly":
            out_kernel_in_dim += out_channels

        # Prepare GNO channel MLP layers
        out_gno_channel_mlp_hidden_layers = gno_config.out_gno_channel_mlp_hidden_layers.copy()
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(in_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            attention_type=gno_config.attention_type,
            coord_dim=self.coord_dim,
            sampling_strategy=gno_config.sampling_strategy,
            max_neighbors = gno_config.max_neighbors,
            sample_ratio=gno_config.sample_ratio,
        )

        self.projection = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=gno_config.projection_channels,
            n_layers=2,
            n_dim=1,
        )

        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=gno_config.gno_coord_dim,
                output_dim=in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1
            )

        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)  

    def forward(self,
                latent_tokens_coord: torch.Tensor, 
                rndata: torch.Tensor,             
                query_coord: torch.Tensor,         
                decoder_nbrs: Optional[List[List[any]]] = None): 
        """
        Forward pass for the MAGNO Decoder.

        Args:
            latent_tokens_coord: Latent grid node coordinates (source). Shape: [num_latent_nodes, coord_dim]
            rndata: Features on the latent grid (source). Shape: [batch_size, num_latent_nodes, in_channels]
            query_coord: Coordinates of the target physical nodes (query points). Shape: [num_query_nodes, coord_dim]
            decoder_nbrs: Optional precomputed neighbor lists for each scale shared with all batch.
                          Required if self.precompute_edges is True.
                          Format: List[neighbors_scale_0, neighbors_scale_1,...]
                          list length is num_scales.

        Returns:
            torch.Tensor: Decoded features on the physical grid. Shape: [batch_size, num_query_nodes, out_channels]
        """
        ## --- Check dimensions ---
        if len(latent_tokens_coord.shape) != 2 or latent_tokens_coord.shape[1] != self.coord_dim:
            raise ValueError(f"Expected latent_tokens_coord shape [num_latent_nodes, {self.coord_dim}], "
                            f"got {latent_tokens_coord.shape}")
        
        num_latent_nodes = latent_tokens_coord.shape[0]
        if len(rndata.shape) != 3 or rndata.shape[1] != num_latent_nodes:
            raise ValueError(f"Expected rndata shape [batch_size, {num_latent_nodes}, in_channels], "
                            f"got {rndata.shape}")
        
        if len(query_coord.shape) != 2 or query_coord.shape[1] != self.coord_dim:
            raise ValueError(f"Expected query_coord shape [num_query_nodes, {self.coord_dim}], "
                            f"got {query_coord.shape}")
        
        if self.precompute_edgs and decoder_nbrs is None:
            raise ValueError("decoder_nbrs is required when precompute_edges is True")
        
        if decoder_nbrs is not None and len(decoder_nbrs) != len(self.scales):
            raise ValueError(f"Expected decoder_nbrs to have length {len(self.scales)} (number of scales), "
                            f"got {len(decoder_nbrs)}")
        # --- End of check dimensions ---   
        
        batch_size = rndata.shape[0]
        device = rndata.device
        # --- 1. Precompute neighbor search  ---
        if self.precompute_edgs:
            self.spatial_nbrs_scales = decoder_nbrs
        else:
            if self.graph_cache is None:
                self.spatial_nbrs_scales = []
                for scale in self.scales:
                    scaled_radius = self.gno_radius * scale
                    spatial_nbrs = self.nb_search(
                        data = latent_tokens_coord,
                        queries = query_coord,
                        radius = scaled_radius
                    )
                    self.spatial_nbrs_scales.append(spatial_nbrs)
                self.graph_cache = True

        # --- 2. Decode features for each scale ---
        decoded_scales = []
        for idx, scale in enumerate(self.scales):
            spatial_nbrs = self.spatial_nbrs_scales[idx]
            decoded = self.gno(
                y = latent_tokens_coord,
                x = query_coord,
                f_y = rndata,
                neighbors = spatial_nbrs
            )

            ## Apply optional geometric embedding
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    input_geom = latent_tokens_coord,
                    latent_queries = query_coord,
                    spatial_nbrs = spatial_nbrs
                )

                geoembedding = geoembedding[None, :, :]
                geoembedding = geoembedding.repeat([batch_size, 1, 1])
                decoded = torch.cat([decoded, geoembedding], dim=-1)
                decoded = decoded.permute(0, 2, 1)
                decoded = self.recovery(decoded).permute(0, 2, 1)
            ## End of optional geometric embedding
            decoded_scales.append(decoded)

        # --- 3. Combine decoded features across scales ---
        if len(decoded_scales) == 1:
            decoded = decoded_scales[0]
        else:
            if self.use_scale_weights:
                scale_weights = self.scale_weighting(query_coord)               # [num_query_points, num_scales]
                scale_weights = self.scale_weight_activation(scale_weights)     # [num_query_points, num_scales]
                scale_weights = scale_weights.permute(1, 0)                     # [num_scales, num_query_points]
                scale_weights = scale_weights.unsqueeze(0).unsqueeze(-1)        # [1, num_scales, num_query_points, 1]
                scale_weights = scale_weights.repeat(batch_size, 1, 1, 1)       # [batch_size, num_scales, num_query_points, 1]
                decoded_scales_tensor = torch.stack(decoded_scales, dim=0)
                decoded_scales_tensor = decoded_scales_tensor.permute(1, 0, 2, 3)
                weighted_decoded_scales = decoded_scales_tensor * scale_weights
                decoded = weighted_decoded_scales.sum(dim=1)                    # [batch_size, num_query_points, feature_dim]
            else:
                decoded = torch.stack(decoded_scales, dim=0).sum(dim=0)

        # --- 4. Project to output channels ---
        decoded = decoded.permute(0, 2, 1)
        decoded = self.projection(decoded).permute(0, 2, 1)

        return decoded