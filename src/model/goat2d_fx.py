import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from .layers.attn import Transformer
from .layers.magno2d_fx import MAGNOEncoder, MAGNODecoder


class GOAT2D_FX(nn.Module):
    """
    Geometry-Aware Operator Transformer (GOAT) for 2D fixed coordinate meshes: 
    Multiscale Attentional Graph Neural Operator (MAGNO) + U Vision Transformer (UViT) + Multiscale Attentional Graph Neural Operator
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: Optional[dataclass] = None):
        nn.Module.__init__(self)
        # --- Define model parameters ---
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = config.args.magno.lifting_channels 
        self.patch_size = config.args.transformer.patch_size
        latent_tokens_size = config.latent_tokens_size
        self.H = latent_tokens_size[0]
        self.W = latent_tokens_size[1]

        ## Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, self.node_latent_size, config.args.magno)
        self.processor = self.init_processor(self.node_latent_size, config.args.transformer)
        self.decoder = self.init_decoder(output_size, self.node_latent_size, config.args.magno)
    
    def init_encoder(self, input_size, latent_size, gno_config):
        return MAGNOEncoder(
            in_channels = input_size,
            out_channels = latent_size,
            gno_config = gno_config
        )
    
    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * node_latent_size,
                                      self.patch_size * self.patch_size * node_latent_size)
    
        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()

        setattr(config.attn_config, 'H', self.H)
        setattr(config.attn_config, 'W', self.W)

        return Transformer(
            input_size=self.node_latent_size * self.patch_size * self.patch_size,
            output_size=self.node_latent_size * self.patch_size * self.patch_size,
            config=config
        )

    def init_decoder(self, output_size, latent_size, gno_config):
        # Initialize the GNO decoder
        return MAGNODecoder(
            in_channels=latent_size,
            out_channels=output_size,
            gno_config=gno_config
        )

    def _get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_H = self.H // self.patch_size
        num_patches_W = self.W // self.patch_size
        positions = torch.stack(torch.meshgrid(
            torch.arange(num_patches_H, dtype=torch.float32),
            torch.arange(num_patches_W, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)

        return positions

    def _compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute absolute embeddings for the given positions.
        """
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb

    def encode(self, x_coord: torch.Tensor, 
               pndata: torch.Tensor, 
               latent_tokens_coord: torch.Tensor, 
               encoder_nbrs: list) -> torch.Tensor:
        
        encoded = self.encoder(
            x_coord = x_coord, 
            pndata = pndata,
            latent_tokens_coord = latent_tokens_coord,
            encoder_nbrs = encoder_nbrs)
        
        return encoded

    def process(self, rndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        rndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        H, W = self.H, self.W
        
        # --- Check the input shape ---
        assert n_regional_nodes == H * W, \
            f"n_regional_nodes ({n_regional_nodes}) is not equal to H ({H}) * W ({W})"
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"H({H}) and W({W}) must be divisible by P({P})"

        # --- Reshape the input data ---
        num_patches_H = H // P
        num_patches_W = W // P
        num_patches = num_patches_H * num_patches_W
        ##  Reshape to patches
        rndata = rndata.view(batch_size, H, W, C)
        rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)
        
        ## --- Apply Vision Transformer processor ---
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)  # shape [num_patches, 2]

        if self.positional_embedding_name == 'absolute':
            pos_emb = self._compute_absolute_embeddings(pos, self.patch_size * self.patch_size * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None
    
        elif self.positional_embedding_name == 'rope':
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        ## --- Reshape back to the original shape ---
        rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, H * W, C)

        return rndata

    def decode(self, latent_tokens_coord: torch.Tensor, 
               rndata: torch.Tensor, 
               query_coord: torch.Tensor, 
               decoder_nbrs: list) -> torch.Tensor:
        
        # Apply MAGNO decoder
        decoded = self.decoder(
            latent_tokens_coord = latent_tokens_coord,
            rndata = rndata, 
            query_coord = query_coord,
            decoder_nbrs = decoder_nbrs)
        
        return decoded

    def forward(self,
                latent_tokens_coord: torch.Tensor,
                xcoord: torch.Tensor,
                pndata: torch.Tensor,
                query_coord: Optional[torch.Tensor] = None,
                encoder_nbrs: Optional[list] = None,
                decoder_nbrs: Optional[list] = None,
                condition: Optional[float] = None,
                ) -> torch.Tensor:
        """
        Forward pass for GIVI model.

        Parameters
        ----------
        latent_tokens_coord: torch.Tensor
            ND Tensor of shape [n_regional_nodes, n_dim]
        xcoord: Optional[torch.Tensor]
            ND Tensor of shape [n_physical_nodes, n_dim]
        pndata: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_physical_nodes, input_size]
        query_coord: Optional[torch.Tensor]
            ND Tensor of shape [n_physical_nodes, n_dim]
        condition: Optional[float]
            The condition of the model
        encoder_nbrs: Optional[list]
            List of neighbors for the encoder
        decoder_nbrs: Optional[list]
            List of neighbors for the decoder

        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, n_physical_nodes, output_size]
        """
        # Encode: Map physical nodes to regional nodes using GNO Encoder
        rndata = self.encode(
            x_coord = xcoord, 
            pndata = pndata,
            latent_tokens_coord = latent_tokens_coord,
            encoder_nbrs = encoder_nbrs)
        
        # Process: Apply Vision Transformer on the regional nodes
        rndata = self.process(
            rndata = rndata, 
            condition = condition)

        # Decode: Map regional nodes back to physical nodes using GNO Decoder
        if query_coord is None:
            query_coord = xcoord
        output = self.decode(
            latent_tokens_coord = latent_tokens_coord,
            rndata = rndata, 
            query_coord = query_coord,
            decoder_nbrs = decoder_nbrs)

        return output
