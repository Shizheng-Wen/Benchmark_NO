# Basics
import os
import torch.nn as nn
import torch
import numpy as np
import xarray as xr
import argparse
from omegaconf import OmegaConf
import gc
import time
from contextlib import nullcontext
import pandas as pd
from copy import deepcopy
import functools
import operator
from pathlib import Path   
import json   


# Dataset
from src.data.dataset import Metadata, DATASET_METADATA
from torch.utils.data import DataLoader, TensorDataset
from src.utils.io_adapter import get_adapter
from src.utils.scale import rescale

# model
from src.model import init_model_from_rigraph
from src.graph import RegionInteractionGraph



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

def get_batch_structure_adapter(dataset):
    """Gets a single sample from the dataset for Max BS finder."""
    if dataset is None or len(dataset) == 0:
         raise ValueError("Dataset is None or empty, cannot get sample structure.")
    return dataset[0] # Returns (c_tensor, u_tensor, x_tensor)

def _make_batch(dummy_sample_structure, bs, device, adapter):
    batch_list = [dummy_sample_structure] * bs
    raw_batch = adapter.collate(batch_list)
    return adapter.to_device(raw_batch, device)

def _prepare_model(model, mode, device):
    train = mode == "training"
    model.train(train).to(device)
    opt = torch.optim.AdamW(model.parameters()) if train else None
    return train, opt

def _make_hashable(df: pd.DataFrame, cols):
    """
    Ensure all key columns are hashable by JSON-dumping dict / list objects.
    Works in-place and returns df for chaining.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            )
    return df

def save_or_update_csv(new_df: pd.DataFrame,
                       csv_path: str | Path,
                       key_cols=("grid", "model_args")):
    """
    Append or update rows, identified by `key_cols`, into CSV at `csv_path`.
    Any dict / list in key columns is JSON-encoded first so it becomes hashable.
    """
    csv_path = Path(csv_path)
    new_df = _make_hashable(new_df.copy(), key_cols)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        old_df = _make_hashable(old_df, key_cols)

        # drop rows with keys that will be replaced
        merged = (old_df.set_index(list(key_cols))
                        .drop(new_df.set_index(list(key_cols)).index, errors="ignore")
                        .reset_index())
        merged = pd.concat([merged, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(csv_path, index=False)
    print(f"[✓] Results written to {csv_path}  (rows={len(merged)})")

def _deep_set(dic, key_path, value, sep = "."):
    keys = key_path.split(sep)
    functools.reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], dic)[keys[-1]] = value

def _recursive_update(target: dict, patch: dict):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _recursive_update(target[k], v)
        else:
            target[k] = v

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
   
    c_array = c_array[0:10]
    x_array = x_array[0:10]
    u_array = u_array[0:10]
    
    ## --- Dataset Specific Handling ---
    if dataset_config.input_grid is not None:
        input_grid = int(dataset_config.input_grid)
        cur_nodes = u_array.shape[2]

        if input_grid <= cur_nodes:
            u_array = u_array[:,:,:input_grid,:]
            if c_array is not None:
                c_array = c_array[:,:,:input_grid,:]
            x_array = x_array[:,:,:input_grid,:]
        
        else:
            pad_nodes = input_grid - cur_nodes
            print(f"[init_dataset] input_grid={input_grid} > dataset nodes={cur_nodes}. "
              f"Padding {pad_nodes} random nodes for benchmarking.")
            
            u_pad = np.random.randn(*u_array.shape[:2], pad_nodes, u_array.shape[-1]).astype(u_array.dtype)
            u_array = np.concatenate([u_array, u_pad], axis=2)

            if c_array is not None:
                c_pad = np.random.randn(*c_array.shape[:2], pad_nodes, c_array.shape[-1]).astype(c_array.dtype)
                c_array = np.concatenate([c_array, c_pad], axis=2)
            
            x_min, y_min = metadata.domain_x[0]
            x_max, y_max = metadata.domain_x[1]

            if metadata.fix_x:
                coords_extra = np.random.uniform(
                              low=[x_min, y_min],
                              high=[x_max, y_max],
                              size=(pad_nodes, 2)
                          ).astype(x_array.dtype)
                coords_extra = coords_extra[None, None, ...]
                coords_extra = np.broadcast_to(coords_extra,
                              (x_array.shape[0], x_array.shape[1], pad_nodes, 2))
            
            else:
                coords_extra = np.random.uniform(
                              low=[x_min, y_min],
                              high=[x_max, y_max],
                              size=(x_array.shape[0], x_array.shape[1], pad_nodes, 2)
                           ).astype(x_array.dtype)
            x_array = np.concatenate([x_array, coords_extra], axis=2)

    
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
    # --- Create DataLoader ---
    phy_domain = metadata.domain_x
    x_min, y_min = phy_domain[0]
    x_max, y_max = phy_domain[1]
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

    print("Dataset initialization complete.")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

def get_input_tensor_adapter(batch, device):
    """
    Extracts the input tensor(s) needed for the model from the batch.
    """
    raise NotImplementedError(f"didn't implement the input tensor adapter")

########################
# Benchmarking Script
########################
# ---------- 1. Largest Batch Size ----------
def _step(model, batch_dict, train, optimizer=None):
    try:
        labels = batch_dict.get("labels", None) 
        model_inputs = {k: v for k, v in batch_dict.items() if k != "labels"}

        with torch.set_grad_enabled(train):
            out = model(**model_inputs)
            if train:
                loss = torch.nn.functional.mse_loss(out.float(), labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False
        raise                        

def find_max_bs(model,
                dummy_sample_structure,
                adapter,
                mode="inference",
                initial_bs=128,
                device="cuda",
                min_bs=1):

    model.to(device)
    was_training = model.training
    model.train(mode == "training")

    optimizer = None if mode == "inference" else torch.optim.AdamW(model.parameters())
    torch.backends.cudnn.benchmark = False

    low, high, best = min_bs, initial_bs, 0
    print(f"  Finding Max BS (mode={mode})...", end="", flush=True)

    while low <= high:
        bs = (low + high) // 2
        print(f" Trying BS={bs}...", end="", flush=True)
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        batch = _make_batch(dummy_sample_structure, bs, device, adapter)
        ok = _step(model, batch, train=(mode == "training"), optimizer=optimizer)
        del batch

        if ok:
            print(" OK.", end="", flush=True)
            best  = bs
            low   = bs + 1
        else:
            print(" OOM.", end="", flush=True)
            high  = bs - 1
            if bs == 1:
                break

    print(f" Found Max Safe BS: {best}")
    model.train(was_training)
    return best

# ---------- 2. Throughput & Peak Memory ----------
def measure_throughput(model,
                       dummy_sample_structure,
                       bs,
                       adapter,
                       mode="inference",
                       device="cuda",
                       warmup_iters=10,
                       measure_iters=50):
    """
    return (samples_per_sec, peak_mem_MB)
    """
    if bs <= 0:
        raise RuntimeError("No viable batch size (OOM at BS=1)")
    print(f"  ▶ Measuring THROUGHPUT (mode={mode}, BS={bs}) ...", flush=True)
    train, optimizer = _prepare_model(model, mode, device)
    batch = _make_batch(dummy_sample_structure, bs, device, adapter)

    # reset peak memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        sync = torch.cuda.synchronize
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
    else:
        sync = lambda d=None: None
        start_evt = end_evt = None

    # warm-up
    for _ in range(warmup_iters):
        _step(model, batch, train=train, optimizer=optimizer)
    sync()

    # start timing
    if start_evt: start_evt.record()
    t0 = time.perf_counter()
    for _ in range(measure_iters):
        _step(model, batch, train=train, optimizer=optimizer)
    sync()
    t1 = time.perf_counter()
    if end_evt:
        end_evt.record(); sync()
        elapsed = start_evt.elapsed_time(end_evt) / 1e3      # → second
    else:
        elapsed = t1 - t0

    sps = bs * measure_iters / elapsed                      # samples / second
    peak = (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None
    print(f"    Done. {sps:,.1f} samp/s, peak mem {peak:.1f} MB", flush=True)
    return sps, peak

# ---------- 3. Latency (BS = 1) ----------
def measure_latency(model,
                    dummy_sample_structure,
                    adapter,
                    mode="inference",
                    device="cuda",
                    warmup_iters=20,
                    measure_iters=100):
    """
    return (avg_latency_ms, peak_mem_MB)
    """
    print(f"  ▶ Measuring LATENCY   (mode={mode}, BS=1) ...", flush=True)
    train, optimizer = _prepare_model(model, mode, device)
    batch = _make_batch(dummy_sample_structure, 1, device, adapter)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        sync = torch.cuda.synchronize
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
    else:
        sync = lambda d=None: None
        start_evt = end_evt = None

    # warm-up
    for _ in range(warmup_iters):
        _step(model, batch, train=train, optimizer=optimizer)
    sync()

    # timing
    if start_evt: start_evt.record()
    t0 = time.perf_counter()
    for _ in range(measure_iters):
        _step(model, batch, train=train, optimizer=optimizer)
    sync()
    t1 = time.perf_counter()
    if end_evt:
        end_evt.record(); sync()
        elapsed = start_evt.elapsed_time(end_evt) / 1e3      # → second
    else:
        elapsed = t1 - t0

    lat_ms = elapsed * 1e3 / measure_iters                   # ms / sample
    peak = (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None
    print(f"    Done. {lat_ms:.2f} ms, peak mem {peak:.1f} MB", flush=True)
    return lat_ms, peak

# ---------- 4. Scalability ----------
def benchmark_scalability(
    base_dataset_cfg,
    base_model_cfg,
    base_graph_cfg,
    grid_sizes: list[int],
    model_cfg_variants: list[dict],
    device = torch.device('cuda'),
    adapter_name = "default",
    warmup_iters = 10,
    measure_iters = 50):

    adapter = get_adapter(adapter_name)
    records, total_runs = [], len(grid_sizes) * len(model_cfg_variants)
    run_idx = 0

    t0_all = time.time()
    
    for g in grid_sizes:
        print(f"\n>>> [Grid={g}] Initializing dataset ...")
        ds_cfg = deepcopy(base_dataset_cfg)
        ds_cfg.input_grid = g

        _, _, _, _, val_ds, _ = init_dataset(ds_cfg)
        # graph build
        coords = val_ds[0][-1]
        rigraph = RegionInteractionGraph.from_point_cloud(
            points = coords.to(device),
            **base_graph_cfg
        )
        adapter = get_adapter(adapter_name)
        adapter = adapter(rigraph)
        
        ref_sample = get_batch_structure_adapter(val_ds)
        

        for m_args in model_cfg_variants:
            run_idx += 1
            tag = f"[{run_idx}/{total_runs}] grid={g} model_args={m_args}"
            print(f"\n=== {tag} ===")
            mdl_cfg = deepcopy(base_model_cfg)
            for k, v in m_args.items():
                if '.' in k:            
                    _deep_set(mdl_cfg, k, v)
                else:                   
                    if isinstance(v, dict):
                        _recursive_update(mdl_cfg.setdefault(k, {}), v)
                    else:
                        mdl_cfg[k] = v
            model = init_model_from_rigraph(
                rigraph = rigraph, 
                input_size = mdl_cfg.input_channels, 
                output_size = mdl_cfg.output_channels,
                drop_edge=mdl_cfg.drop_edge,
                variable_mesh=mdl_cfg.variable_mesh,
                model = mdl_cfg.name,
                config = mdl_cfg.args.deepgnn)
            
            n_params = sum([p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]) / 1e6
            print(f"-> Model ready ({n_params:.2f} M params)")

            try:
            # --- Largest BS ---
                max_inf_bs = find_max_bs(model, ref_sample, adapter=adapter,
                                        mode="inference", device=device,
                                        initial_bs=2000, min_bs=1)
                max_trn_bs = find_max_bs(model, ref_sample, adapter=adapter,
                                        mode="training",  device=device,
                                        initial_bs=1000, min_bs=1)
                print(f"   Max BS  | inf {max_inf_bs} | train {max_trn_bs}")
                
                inf_sps, inf_peak = measure_throughput(model, ref_sample, max_inf_bs,
                                                    mode="inference", device=device,
                                                    adapter=adapter,
                                                    warmup_iters=warmup_iters,
                                                    measure_iters=measure_iters)
                trn_sps, trn_peak = measure_throughput(model, ref_sample, max_trn_bs,
                                                    mode="training",  device=device,
                                                    adapter=adapter,
                                                    warmup_iters=warmup_iters,
                                                    measure_iters=measure_iters)
                inf_lat, inf_lat_peak = measure_latency(model, ref_sample,
                                                        mode="inference", device=device,
                                                        adapter=adapter,
                                                        warmup_iters=warmup_iters,
                                                        measure_iters=measure_iters)
                trn_lat, trn_lat_peak = measure_latency(model, ref_sample,
                                                        mode="training",  device=device,
                                                        adapter=adapter,
                                                        warmup_iters=warmup_iters,
                                                        measure_iters=measure_iters)
                
                records.append({
                    "grid": g,
                    "model_args": m_args,
                    "params_M": round(n_params, 2),

                    "max_inf_bs":   max_inf_bs,
                    "max_trn_bs":   max_trn_bs,

                    "inf_sps":      inf_sps,
                    "trn_sps":      trn_sps,

                    "inf_peak_MB":  inf_peak,
                    "trn_peak_MB":  trn_peak,

                    "inf_lat_ms":   inf_lat,
                    "trn_lat_ms":   trn_lat,

                    "inf_bs1_peak_MB": inf_lat_peak,
                    "trn_bs1_peak_MB": trn_lat_peak,
                })
            except RuntimeError as e:
                print(f"   !! {e}. Marking as OOM.")
                from math import nan
                records.append({
                    "grid": g,
                    "model_args": m_args,
                    "params_M": n_params,
                    "max_inf_bs": 0,
                    "max_trn_bs": 0,
                    "inf_sps": nan, "trn_sps": nan,
                    "inf_peak_MB": nan, "trn_peak_MB": nan,
                    "inf_lat_ms": nan, "trn_lat_ms": nan,
                    "inf_bs1_peak_MB": nan, "trn_bs1_peak_MB": nan,
                })
            finally:
                torch.cuda.synchronize()
                del model
                torch.cuda.empty_cache(); gc.collect()
                print(f"<✓> {tag} finished.\n")
        
    elapsed_all = time.time() - t0_all
    print(f"\n### Scalability sweep done in {elapsed_all/60:.1f} min ###")

    return pd.DataFrame.from_records(records)

########################
# Sweep Registry
########################
SWEEPS = {
    "rigno": {
        "model_scale": {
            "grid_sizes": [16431],
            "model_variants": [
                {
                    "args.deepgnn.mpconfig.edge_fn_config.hidden_size": 64,
                    "args.deepgnn.mpconfig.node_fn_config.hidden_size": 64,
                    },
                {
                    "args.deepgnn.mpconfig.edge_fn_config.hidden_size": 128,
                    "args.deepgnn.mpconfig.node_fn_config.hidden_size": 128,
                    },
                {
                    "args.deepgnn.mpconfig.edge_fn_config.hidden_size": 256,
                    "args.deepgnn.mpconfig.node_fn_config.hidden_size": 256,
                    },
                {
                    "args.deepgnn.mpconfig.edge_fn_config.hidden_size": 512,
                    "args.deepgnn.mpconfig.node_fn_config.hidden_size": 512,
                    },
                {
                    "args.deepgnn.mpconfig.edge_fn_config.hidden_size": 1024,
                    "args.deepgnn.mpconfig.node_fn_config.hidden_size": 1024,
                    },
            ],
        },
        "input_scale": {
            "grid_sizes": [1000, 10000, 50000],
            "model_variants": [
                {
                    "args.deepgnn.mpconfig.edge_fn_config.hidden_size": 128,
                    "args.deepgnn.mpconfig.node_fn_config.hidden_size": 128,
                    },
            ],
        },
    },
}

########################
# Main workflow 
########################
def main(config_path, run_scalability=False):
    # --- 1. Load Config ---
    ## File Paths
    data_path = os.path.join(os.path.dirname(__file__), "config/setup_config.json")
    model_path = os.path.join(os.path.dirname(__file__), config_path)
    setup_config = OmegaConf.load(data_path)
    config = OmegaConf.load(config_path)
    
    basic_config = setup_config.basics
    dataset_config = setup_config.dataset
    model_config = config.model
    graph_config = config.graph
    adapter_name = model_config.get("adapter", "default")
    _, _, _, _, val_ds, _ = init_dataset(dataset_config)
    coord = val_ds[0][2]

    adapter = get_adapter(adapter_name)

    results = {
               'config_setup': OmegaConf.to_container(basic_config, resolve=True),
               'config_model': OmegaConf.to_container(model_config, resolve=True),
               'metrics': {}}
    

    # --- 2. environment setup ---
    device = torch.device(basic_config.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA device specified but CUDA is not available. Switching to CPU.")
            device = torch.device('cpu')
            results['config_setup']['device'] = 'cpu'
        else:
            gpu_index = device.index if device.index is not None else torch.cuda.current_device()
            print(f"  GPU Name: {torch.cuda.get_device_name(gpu_index)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3):.2f} GB")

    print(f"\n[1] Initializing model and reference dataset: {dataset_config.name}...")
    
    if run_scalability:
        model_name = model_config.name
        if model_name not in SWEEPS:
            raise ValueError(f"No sweep config for model '{model_name}'")
        
        all_records = []
        
        for sweep_tag, sweep_cfg in SWEEPS[model_name].items():
            print(f"\n[3] Running sweep: {sweep_tag} ...")

            df_scal = benchmark_scalability(dataset_config, 
                                            model_config, 
                                            graph_config,
                                            grid_sizes = sweep_cfg["grid_sizes"],
                                            model_cfg_variants = sweep_cfg["model_variants"],
                                            device = device,
                                            adapter_name=adapter_name,
                                            warmup_iters=5,
                                            measure_iters=10)
            df_scal["sweep"] = sweep_tag
            all_records.append(df_scal)

        df_final = pd.concat(all_records, ignore_index=True)

        df_path = Path("results") / f"{model_name}.csv"
        save_or_update_csv(
            new_df = df_final, 
            csv_path = df_path)
        print(f"\nScalability sweeps finished. Results saved to {df_path}")
    
    else:
        # --- 3. Init Dataset ---
        _, _, _, _, val_ds, _ = init_dataset(dataset_config)
        
        # --- 4. Graph Build ---
        coords = val_ds[0][-1]
        rigraph = RegionInteractionGraph.from_point_cloud(
            points = coords.to(device),
            **graph_config
        )
        adapter = adapter(rigraph)

        # --- 5. Init Model ---
        print(f"Initializing model: {model_config.name}")
        model = init_model_from_rigraph(
            rigraph = rigraph, 
            input_size = model_config.input_channels, 
            output_size = model_config.output_channels,
            drop_edge=model_config.drop_edge,
            variable_mesh=model_config.variable_mesh,
            model = model_config.name,
            config = model_config.args.deepgnn)
        
        nparam = sum(
            [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
            ) / 1e6
        nbytes = sum(
            [p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size() for p in model.parameters()]
            )
        print(f"-> Model ready ({nparam:.2f} M params)")

        # --- 5. Mersure the performance ---
        ref_dummy_sample = get_batch_structure_adapter(val_ds)
        max_inf_bs = find_max_bs(model, ref_dummy_sample, adapter=adapter,
                                mode='inference', device=device, initial_bs=2000, min_bs=64)
        max_train_bs = find_max_bs(model, ref_dummy_sample, adapter=adapter,
                                mode='training', device=device, initial_bs=1000, min_bs=1)

        print("\n[2] Benchmarking throughput / latency...\n")
        ## --- Throughput @ Max BS ---
        inf_tput, inf_peak = measure_throughput(model, ref_dummy_sample, max_inf_bs, adapter=adapter,
                                                mode="inference", device=device)
        train_tput, train_peak = measure_throughput(model, ref_dummy_sample, max_train_bs, adapter=adapter,
                                                    mode="training", device=device)

        ## --- BS = 1 Latency ---
        inf_lat, inf_lat_peak = measure_latency(model, ref_dummy_sample, adapter=adapter,
                                                mode="inference", device=device)
        train_lat, train_lat_peak = measure_latency(model, ref_dummy_sample, adapter=adapter,
                                                    mode="training", device=device)

        print(f"★ Inference:\n"
            f"    • Max-BS = {max_inf_bs:4d} → {inf_tput:8.1f} samp/s, "
            f"peak mem {inf_peak:.1f} MB\n"
            f"    • BS = 1         → {inf_lat:8.2f} ms / sample, "
            f"peak mem {inf_lat_peak:.1f} MB")

        print(f"★ Training:\n"
            f"    • Max-BS = {max_train_bs:4d} → {train_tput:8.1f} samp/s, "
            f"peak mem {train_peak:.1f} MB\n"
            f"    • BS = 1         → {train_lat:8.2f} ms / step, "
            f"peak mem {train_lat_peak:.1f} MB")

        results["metrics"].update({
            "nparam": round(nparam, 2),
            "max_inf_bs":   max_inf_bs,
            "max_train_bs": max_train_bs,
            "throughput_inf_sps":   inf_tput,
            "throughput_train_sps": train_tput,
            "latency_inf_ms":   inf_lat,
            "latency_train_ms": train_lat,
            "peak_mem_inf_mb":  inf_peak,
            "peak_mem_train_mb":train_peak,
        })
        save_or_update_csv(pd.DataFrame([results["metrics"]]), f"results/test/{model_config.name}.csv")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Script")
    parser.add_argument("-c", "--config", type=str, 
                        default="config/model_config/rigno.json", 
                        help="config file path")
    parser.add_argument("--scalability", action="store_true",
                        help="run scalability sweep (input-grid × model-size)")
    args = parser.parse_args()
    main(args.config, run_scalability=args.scalability)