import torch,gc
from abc import ABC, abstractmethod
from .scale import rescale
import copy

def custom_collate_fn(batch):
    """collates data points with coordinates"""
    inputs = torch.stack([item[0] for item in batch])          
    labels = torch.stack([item[1] for item in batch])         
    coords = torch.stack([item[2] for item in batch])          
    return inputs, labels, coords

class IOAdapter(ABC):
    @abstractmethod
    def collate(self, batch_list):
     """ sample group -> batch structure"""
    
    @abstractmethod
    def to_device(self, batch, device):
        """"put batch into device"""

class DefaultAdapter(IOAdapter):
    def collate(self, batch_list):
        inputs, labels, coords = custom_collate_fn(batch_list)
        return {"inputs": inputs, "labels": labels, "coords": coords}

    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class TransolverAdapter(IOAdapter):
    def collate(self, batch_list):
        inputs, labels, coords = custom_collate_fn(batch_list)
        return {"fx": inputs, "labels": labels, "x": coords}

    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class GoatAdapter(IOAdapter):
    def __init__(self, phy_domain = ([-.5, -.5], [1.5, 1.5]), token_size = (64, 64)):
        super().__init__()
        x_min, y_min = phy_domain[0]
        x_max, y_max = phy_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, token_size[0]), 
            torch.linspace(y_min, y_max, token_size[1]), 
            indexing='ij' 
        )
        self.latent_tokens_coord = torch.stack(meshgrid, dim=-1).reshape(-1,2)
        self.latent_tokens_coord = rescale(self.latent_tokens_coord, (-1,1), phy_domain)

    def collate(self, batch_list):
        inputs, labels, coords = custom_collate_fn(batch_list)
        return {"latent_tokens_coord": self.latent_tokens_coord, "xcoord": coords[0], "pndata": inputs, "labels": labels}
    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class GINOAdapter(IOAdapter):
    def __init__(self, phy_domain = ([-.5, -.5], [1.5, 1.5]), token_size = (64, 64)):
        super().__init__()
        x_min, y_min = phy_domain[0]
        x_max, y_max = phy_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, token_size[0]), 
            torch.linspace(y_min, y_max, token_size[1]), 
            indexing='ij' 
        )
        self.latent_queries = torch.stack(meshgrid, dim=-1).unsqueeze(0)
        self.latent_queries = rescale(self.latent_queries)
    def collate(self, batch_list):
        inputs, labels, coords = custom_collate_fn(batch_list)
        return {"x": inputs, "input_geom": coords[0:1], "latent_queries": self.latent_queries, "output_queries": coords[0:1],"labels": labels}
    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class GeoFNOAdapter(IOAdapter):
    def collate(self, batch_list):
        """
        Collates data and prepares the single input tensor 'x'
        for GeoFNO, combining input features (if available) and coordinates.
        Input 'x' shape: (batch, num_points, num_features + 2)
        """
        inputs, labels, coords = custom_collate_fn(batch_list)
    
        if inputs is not None:
            assert inputs.shape[:-1] == coords.shape[:-1], \
                f"Input features shape {inputs.shape[:-1]} must match coordinate shape {coords.shape[:-1]}"
            model_input_x = torch.cat((inputs, coords), dim=-1)
        else:
            print("Warning: No input features ('c') found in batch. Using only coordinates as input 'x' for GeoFNO.")
            model_input_x = coords

        return {"x": model_input_x, "labels": labels}

    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class FNODSEAdapter(IOAdapter):
    def collate(self, batch_list):
        """
        Collates data and prepares the single input tensor 'positions'
        for FNO_dse, combining input features (if available) and coordinates.
        Input 'positions' shape: (batch, num_points, num_features + 2)
        """
        inputs, labels, coords = custom_collate_fn(batch_list)

        if inputs is not None:
            assert inputs.shape[:-1] == coords.shape[:-1], \
                f"Input features shape {inputs.shape[:-1]} must match coordinate shape {coords.shape[:-1]}"
            model_input_positions = torch.cat((coords, inputs), dim=-1)
        else:
            print("Warning: No input features ('c') found in batch. Using only coordinates as input 'positions' for FNO_dse.")
            model_input_positions = coords 

        return {"positions": model_input_positions, "labels": labels}

    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class RIGNOAdapter(IOAdapter):
    def __init__(self, rigraph):
        super().__init__()
        self.rigraph = rigraph

    def collate(self, batch_list):
        inputs, labels, coords = custom_collate_fn(batch_list)
        graph = copy.deepcopy(self.rigraph)
        
        return {"graphs": graph, "pndata": inputs, "labels": labels}

    def to_device(self, batch, device):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

_ADAPTERS = {
    "default": DefaultAdapter(),
    "transolver": TransolverAdapter(),
    "goat": GoatAdapter(),
    "gino": GINOAdapter(),
    "geofno": GeoFNOAdapter(),
    "fnodse": FNODSEAdapter(),
    "rigno": RIGNOAdapter
    }

def register_adapter(name: str, adapter: IOAdapter):
    _ADAPTERS[name] = adapter

def get_adapter(name: str):
    return _ADAPTERS.get(name, _ADAPTERS["default"])
