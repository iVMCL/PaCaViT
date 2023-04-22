# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (
    ConfigType,
    OptConfigType,
    OptMultiConfig,
    OptSampleList,
    SampleList,
    add_prefix,
)

from mmseg.models.segmentors import EncoderDecoder


@MODELS.register_module()
class PaCaEncoderDecoder(EncoderDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _decode_head_forward_train(
        self, inputs: List[Tensor], data_samples: SampleList
    ) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode[0], "decode"))
        losses.update(add_prefix(loss_decode[1], "paca"))
        return losses
