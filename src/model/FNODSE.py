import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


################################################################
# FNO_dse (VFT, SpectralConv2d_dse, FNO_dse same as Elasticity)
################################################################
def add_padding(points: torch.Tensor, padding_ratio=0.2) -> torch.Tensor:
    """
    Add zero padding to the point cloud.
    Assumes points are in [0, 1] range for x and y.
    Args:
        points (torch.Tensor): [B, N, 5] tensor, with x in [:, :, 0] and y in [:, :, 1].
        padding_ratio (float): proportion of the original range to pad.
    Returns:
        torch.Tensor: padded point cloud
    """
    scale = 1.0 - padding_ratio * 2
    offset = padding_ratio
    padded = points.clone()
    padded[:, :, 0] = padded[:, :, 0] * scale + offset  # x
    padded[:, :, 1] = padded[:, :, 1] * scale + offset  # y
    return padded

def remove_padding(padded_points: torch.Tensor, padding_ratio=0.2) -> torch.Tensor:
    """
    Remove zero padding from the point cloud.
    Args:
        padded_points (torch.Tensor): [N, 5] tensor, with x in [:, 0] and y in [:, 1].
        padding_ratio (float): proportion of the original range used as padding.
    Returns:
        torch.Tensor: original (unpadded) point cloud
    """
    scale = 1.0 / (1.0 - padding_ratio * 2)
    offset = padding_ratio
    unpadded = padded_points.clone()
    unpadded[:, 0] = (unpadded[:, 0] - offset) * scale  # x
    unpadded[:, 1] = (unpadded[:, 1] - offset) * scale  # y
    return unpadded

# class for fully nonequispaced 2d points
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        device = x_positions.device
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().to(device)
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().to(device)


        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat)) 

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)
        
        return data_inv
    

class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_dse, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, transformer):
        batchsize = x.shape[0]
        num_pts = x.shape[-1]

        x = x.permute(0, 2, 1)
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = x_ft.permute(0, 2, 1)
        # out_ft = self.compl_mul1d(x_ft, self.weights3)
        x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, 2*self.modes1, 2*self.modes1-1))

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights2)

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1**2))
        x_ft2 = x_ft[..., 2*self.modes1:].flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real


class FNO_dse(nn.Module):
    # Set a class attribute for the default configs.
    configs = {
        'num_train':            1024,
        'num_val':              256,
        'num_test':             256,
        'batch_size':           20, 
        'epochs':               10,
        'test_epochs':          1,

        'datapath':             "/cluster/work/math/camlab-data/rigno-unstructured/",  # Path to data
        'data_small_domain':    True,              # Whether to use a small domain or not for specifically the Airfoil experiment

        # Training specific parameters
        'learning_rate':        0.001,
        'scheduler_step':       50,
        'scheduler_gamma':      0.5,
        'weight_decay':         1e-4,                   # Weight decay
        'loss_fn':              'L1',                   # Loss function to use - L1, L2

        # Model specific parameters
        'modes1':               12,                     # Number of x-modes to use in the Fourier layer
        'modes2':               12,                     # Number of y-modes to use in the Fourier layer
        'width':                32,                     # Number of channels in the convolutional layers
    }
    def __init__(self, configs):
        super(FNO_dse, self).__init__()

        self.modes1 = configs['modes1']
        self.modes2 = configs['modes2']
        self.width = configs['width']
        self.padding = 0 # pad the domain if input is non-periodic

        
        self.fc0 = nn.Linear(configs['in_channels'], self.width)

        self.conv0 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.b0 = nn.Conv1d(configs['in_channels'], self.width, 1)
        self.b1 = nn.Conv1d(configs['in_channels'], self.width, 1)
        self.b2 = nn.Conv1d(configs['in_channels'], self.width, 1)
        self.b3 = nn.Conv1d(configs['in_channels'], self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, configs['out_channels'])


    def forward(self, positions):
        positions = add_padding(positions)
        transform = VFT(positions[:,:,0], positions[:,:,1], self.modes1)
        x = positions

        grid = positions.permute(0,2,1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x, transform)
        x2 = self.w0(x)
        x3 = self.b0(grid)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv1(x, transform)
        x2 = self.w1(x)
        x3 = self.b1(grid)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv2(x, transform)
        x2 = self.w2(x)
        x3 = self.b2(grid)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv3(x, transform)
        x2 = self.w3(x)
        x3 = self.b3(grid)
        x = x1 + x2 + x3
        
        x = remove_padding(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)

        return x