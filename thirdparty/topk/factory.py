import os
from functools import partial
from pathlib import Path
from typing import Sequence, Tuple, Union, Callable, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from dinov2.factory import dinov2_factory
from dinov2.layers import MemEffAttention
from dinov2.layers import NestedTensorBlock as Block

from .dinov2 import dinov2_apply_patch

def dinov2_topk_factory(
    dinov2_model_name: Literal[ "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    r: Union[List[int], int],
    **kwargs
) -> Tuple:

    ### Get the baseline dinov2 model
    dinov2_model, dinov2_config = dinov2_factory(dinov2_model_name)

    ### Now, wrap it with tome
    dinov2_apply_patch(dinov2_model)

    ### Set r accordingly
    dinov2_model.r = r

    return (
        dinov2_model,
        dinov2_config
    )


if __name__ == "__main__":
    model, conf = dinov2_topk_factory("dinov2_vits14", r=2)