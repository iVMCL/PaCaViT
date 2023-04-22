from itertools import islice
from typing import Optional, Callable, Tuple
import json

import torch
import torch.nn as nn

from timm.optim.optim_factory import _layer_map


# modified from timm.optim.optim_factory.param_groups_layer_decay
def layerwise_lr_decay(
    model: nn.Module,
    num_groups=12,
    weight_decay: float = 0.05,
    no_weight_decay_list: Tuple[str] = (),
    layer_decay: float = 0.75,
    _logger=None,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    Based on BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    layer_map = _layer_map(model, num_groups=num_groups)

    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    if _loger is not None:
        _loger.info(f"\n{json.dumps(param_group_names, indent=2)}\n")

    return list(param_groups.values())
