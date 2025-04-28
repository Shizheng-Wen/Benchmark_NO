import torch.nn as nn
import torch
from einops import rearrange
import numpy as np
import math

def unified_pos_embedding(shapelist, ref, batchsize=1):
    if len(shapelist) == 1:
        size_x = shapelist[0]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1]).cuda()  # B N 1
        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        grid_ref = gridx.reshape(1, ref, 1).repeat([batchsize, 1, 1]).cuda()  # B N 1
        pos = torch.sqrt(torch.sum((grid[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, ref).contiguous()
    if len(shapelist) == 2:
        size_x, size_y = shapelist[0], shapelist[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x * size_y, ref * ref).contiguous()
    if len(shapelist) == 3:
        size_x, size_y, size_z = shapelist[0], shapelist[1], shapelist[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).cuda()  # B H W D 3

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1, 1).repeat([batchsize, 1, ref, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1, 1).repeat([batchsize, ref, 1, ref, 1])
        gridz = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, ref, 1).repeat([batchsize, ref, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda()  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, size_x * size_y * size_z, ref * ref * ref).contiguous()
    return pos

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:,:1])], dim=-1)
    return embedding
