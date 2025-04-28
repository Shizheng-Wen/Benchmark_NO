# Basics
import os
import torch.nn as nn
import torch
import numpy as np
import xarray as xr
import argparse
from omegaconf import OmegaConf

# Dataset
from src.data.dataset import Metadata, DATASET_METADATA
from torch.utils.data import DataLoader, TensorDataset




#######################
# Utils
#######################
EPSILON = 1e-10

def custom_collate_fn(batch):
    """collates data points with coordinates"""
    inputs = torch.stack([item[0] for item in batch])          
    labels = torch.stack([item[1] for item in batch])         
    coords = torch.stack([item[2] for item in batch])          
    return inputs, labels, coords


#######################
# Init Dataset
#######################
def init_dataset(dataset_config):
    # --- 1. Load, Split, and Normalize Data ---    
    print("Loading and preprocessing data...")
    base_path = dataset_config.base_path
    dataset_name = dataset_config.name
    dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
    metadata = DATASET_METADATA[dataset_config.metaname]
    
    ## --- Load Dataset ---
    poseidon_dataset_name = ["Poisson-Gauss", "SE-AF"]
    with xr.open_dataset(dataset_path) as ds:
        u_array = ds[metadata.group_u].values           # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
        if metadata.group_c is not None:
            c_array = ds[metadata.group_c].values       # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
        else:
            c_array = None
        if metadata.group_x is not None and metadata.fix_x == True:
            x_array = ds[metadata.group_x].values      # Shape: [1, num_timesteps, num_nodes, num_dims]
            x_array = np.repeat(x_array, u_array.shape[0], axis=0) # Shape: [num_samples, num_timesteps, num_nodes, num_dims]
        elif metadata.group_x is not None and metadata.fix_x == False:
            x_array = ds[metadata.group_x].values     # Shape: [num_samples, num_timesteps, num_nodes, num_dims]
        else:
            domain_x = metadata.domain_x               # Shape: ([x_min, y_min], [x_max, y_max])
            nx, ny = u_array.shape[-2], u_array.shape[-1]
            x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
            y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
            xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
            x_array = np.stack((xv, yv), axis=-1)               # Shape: [num_nodes, 2]
            x_array = x_array.reshape(-1, 2)                    # Shape: [num_nodes, 2]
            num_nodes = x_array.shape[0]
            num_samples = u_array.shape[0]
            
            c_array = c_array.reshape(num_samples,-1, c_array.shape[-1])if c_array is not None else None
            u_array = u_array.reshape(num_samples,-1, u_array.shape[-1])
            node_permutation = np.random.permutation(num_nodes)
            x_array = x_array[node_permutation, :]
            u_array = u_array[:, node_permutation, :]
            if c_array is not None:
                c_array = c_array[:, node_permutation, :]
            x_array = x_array[np.newaxis, np.newaxis, :, :]             # Shape: [1, 1, num_nodes, 2]
            x_array = np.repeat(x_array, num_samples, axis=0)
            u_array = u_array[:, np.newaxis, :, :]                     # Shape: [num_samples, 1, num_nodes, num_channels]
            if c_array is not None:
                c_array = c_array[:, np.newaxis, :, :]                 # Shape: [num_samples, 1, num_nodes, num_channels_c]

    ## --- Dataset Specific Handling ---
    if dataset_config.input_grid is not None:
        input_grid = dataset_config.input_grid
        u_array = u_array[:,:,:input_grid,:]
        if c_array is not None:
            c_array = c_array[:,:,:input_grid,:]
        x_array = x_array[:,:,:input_grid,:]
    
    ## --- Select Variables & Check Shapes ---
    active_vars = metadata.active_variables
    u_array = u_array[..., active_vars]
    num_input_channels = c_array.shape[-1] if c_array is not None else 0
    num_output_channels = u_array.shape[-1]

    ## --- Compute Sizes & Indices ---
    total_samples = u_array.shape[0]
    train_size = dataset_config.train_size
    val_size = dataset_config.val_size
    test_size = dataset_config.test_size
    assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
    assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."

    if dataset_config.rand_dataset:
        indices = np.random.permutation(len(u_array))
    else:
        indices = np.arange(len(u_array))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[-test_size:]
    # Split data into train, val, test
    u_train = np.ascontiguousarray(u_array[train_indices])
    u_val = np.ascontiguousarray(u_array[val_indices])
    u_test = np.ascontiguousarray(u_array[test_indices])
    x_train = np.ascontiguousarray(x_array[train_indices])
    x_val = np.ascontiguousarray(x_array[val_indices])
    x_test = np.ascontiguousarray(x_array[test_indices])

    if c_array is not None:
        c_train = np.ascontiguousarray(c_array[train_indices])
        c_val = np.ascontiguousarray(c_array[val_indices])
        c_test = np.ascontiguousarray(c_array[test_indices])
    else:
        c_train = c_val = c_test = None
    # --- Compute Statics & Normalize (using training set only) ---
    print("Computing statistics and normalizing data")
    print("You need to make sure that the u_mean and u_std are the same for testing other datasets!")
    u_train_flat = u_train.reshape(-1, u_train.shape[-1])
    u_mean = np.mean(u_train_flat, axis=0)
    u_std = np.std(u_train_flat, axis=0) + EPSILON
    u_train = (u_train - u_mean) / u_std
    u_val = (u_val - u_mean) / u_std
    u_test = (u_test - u_mean) / u_std
    c_mean = None
    c_std = None
    if c_array is not None:
        c_train_flat = c_train.reshape(-1, c_train.shape[-1])
        c_mean = np.mean(c_train_flat, axis=0)
        c_std = np.std(c_train_flat, axis=0) + EPSILON
        c_train = (c_train - c_mean) / c_std
        c_val = (c_val - c_mean) / c_std
        c_test = (c_test - c_mean) / c_std
        c_train = torch.tensor(c_train, dtype=torch.float32).squeeze(1)
        c_val = torch.tensor(c_val, dtype=torch.float32).squeeze(1)
        c_test = torch.tensor(c_test, dtype=torch.float32).squeeze(1)
    # --- Convert to Tensors ---
    # Handle None case for c_train/val/test when converting
    u_train = torch.tensor(u_train, dtype=torch.float32).squeeze(1)
    u_val = torch.tensor(u_val, dtype=torch.float32).squeeze(1)
    u_test = torch.tensor(u_test, dtype=torch.float32).squeeze(1)
    x_train = torch.tensor(x_train, dtype=torch.float32).squeeze(1)
    x_val = torch.tensor(x_val, dtype=torch.float32).squeeze(1)
    x_test = torch.tensor(x_test, dtype=torch.float32).squeeze(1)
    if c_array is not None:
        c_train = torch.tensor(c_train, dtype=torch.float32).squeeze(1)
        c_val = torch.tensor(c_val, dtype=torch.float32).squeeze(1)
        c_test = torch.tensor(c_test, dtype=torch.float32).squeeze(1)
    else:
        c_train = c_val = c_test = None
    # --- Create DataLoader ---
    train_ds = TensorDataset(c_train, u_train, x_train)
    val_ds = TensorDataset(c_val, u_val, x_val)
    test_ds = TensorDataset(c_test, u_test, x_test)
    train_loader = DataLoader(
        train_ds,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        collate_fn = custom_collate_fn,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
        
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        collate_fn = custom_collate_fn,
        num_workers=dataset_config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        collate_fn = custom_collate_fn,
        num_workers=dataset_config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def get_input_tensor_adapter(batch, device):
    """
    Extracts the input tensor(s) needed for the model from the batch.
    """
    pass

#######################
# Init Model
#######################
def init_model(model_config):
    model = init_model(
        input_size = model_config.input_channels,
        output_size = model_config.output_channels,
        model = model_config.name,
        config = model_config.args
    )
    return model

########################
# Benchmarking Script
########################


def main(config_path):
    # --- 1. Load Config ---
    ## File Paths
    data_path = os.path.join(os.path.dirname(__file__), "config/data_config.json")
    dataset_config = OmegaConf.load(data_path)
    config = OmegaConf.load(config_path)
    model_config = config.model

    # --- 2. Init Dataset ---
    train_loader, val_loader, test_loader = init_dataset(dataset_config)

    # --- 3. Init Model ---
    model = init_model(model_config)

    

if __name__ == "main":
    parser = argparse.ArgumentParser(description="Benchmarking Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to the config file"
    )
    args = parser.parse_args()
    
    main(args.config_path)