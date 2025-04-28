from dataclasses import dataclass
from typing import Tuple, Optional, Union
from omegaconf import OmegaConf

from src.models import Transolver, GNOT

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
        "gnot"
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
    else:
        raise ValueError(f"model {config} not supported currently!")

