from dataclasses import dataclass
from typing import Tuple, Optional, Union
from omegaconf import OmegaConf

from src.model import Transolver, GNOT
from .goat2d_fx import GOAT2D_FX
from .GINO import GINO
from .GeoFNO import Geo_FNO
from .FNODSE import FNO_dse
from .RIGNO import Physical2Regional2Physical
from src.graph.rigraph import RegionInteractionGraph



def merge_config(default_config_class, user_config):
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)

def init_model(
        input_size:int = None, 
        output_size:int = None, 
        model:str = "transolver", 
        config:Optional[dataclass] = None
                ):
    supported_models = [
        "transolver",
        "gnot",
        "goat2d_fx",
        "gino",
        "geofno",
        "fno_dse",
        "rigno"
    ]

    assert model.lower() in supported_models, (
        f"model {model} not supported, only support {supported_models} "
    )

    if model.lower() == "transolver":
        transolverconfig = merge_config(Transolver.ModelConfig, config)
        return Transolver.Model(transolverconfig)
    elif model.lower() == "gnot":
        gnotconfig = merge_config(GNOT.ModelConfig, config)
        return GNOT.Model(gnotconfig)
    elif model.lower() == "goat2d_fx":
        return GOAT2D_FX(
            input_size = input_size,
            output_size = output_size,
            config = config
        )
    elif model.lower() == "gino":
        return GINO(
            in_channels = input_size,
            out_channels = output_size,
            **config
        )
    elif model.lower() == "geofno":
        return Geo_FNO(
            configs = config
        )
    elif model.lower() == "fno_dse":
        return FNO_dse(
            configs = config
        )
    else:
        raise ValueError(f"model {model} not supported currently!")


def init_model_from_rigraph(rigraph:RegionInteractionGraph,
                            input_size: int,
                            output_size: int,
                            drop_edge: float = 0.0,
                            variable_mesh: bool = False,
                            model: str = "rigno",
                            config:dataclass = None
                            ):
    supported_models = [
        "rigno",
    ]

    assert model.lower() in supported_models, (
        f"model {model} not supported, only support {supported_models} "
    )

    if model.lower() == "rigno":
        return Physical2Regional2Physical(
            input_size = input_size, 
            output_size = output_size, 
            rigraph = rigraph,
            drop_edge = drop_edge,
            variable_mesh = variable_mesh,
            config = config)