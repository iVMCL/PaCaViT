from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

from mmseg.registry import MODELS
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.utils import ConfigType, SampleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


@MODELS.register_module()
class PaCaSegHead(BaseDecodeHead):
    """The Patch-to-Cluster Attention head for semantic segmentation

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(
        self,
        interpolate_mode="bilinear",
        aux_loss_decode=dict(
            type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
        ),
        **kwargs,
    ):
        super().__init__(input_transform="multiple_select", **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )

        self.q = nn.Sequential(
            ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            Rearrange("B C H W -> B (H W) C"),
        )

        self.clustering = nn.Sequential(
            ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            nn.Conv2d(
                self.channels, self.num_classes, kernel_size=1
            ),  # TODO: bias=False
        )

        self.k = nn.Sequential(
            nn.Linear(self.channels * num_inputs, self.channels),
            Rearrange("B M C -> B C M"),
            nn.SyncBatchNorm(self.channels),
            Rearrange("B C M -> B M C"),
            build_activation_layer(self.act_cfg),
        )
        self.v = nn.Sequential(
            nn.Linear(self.channels * num_inputs, self.channels),
            Rearrange("B M C -> B C M"),
            nn.SyncBatchNorm(self.channels),
            Rearrange("B C M -> B M C"),
            build_activation_layer(self.act_cfg),
        )
        self.proj = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        if isinstance(aux_loss_decode, dict):
            self.aux_loss_decode = MODELS.build(
                aux_loss_decode
            )  # build_loss(aux_loss_decode)
        else:
            raise TypeError(
                f"aux_loss_decode must be a dict,\
                but got {type(aux_loss_decode)}"
            )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                )
            )

        x = torch.cat(outs, dim=1)
        H, W = x.shape[2:]

        q = self.q(x)  # B N C

        c_raw = self.clustering(x)  # B M H W
        c = rearrange(c_raw, "B M H W -> B M (H W)")
        c = c.softmax(dim=-1)

        x_ = rearrange(x, "B C H W -> B (H W) C")
        z = c @ x_  # B M C
        k = self.k(z)
        v = self.v(z)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        out = attn @ v  # B N C
        out = rearrange(out, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        out = self.proj(out)

        out = self.cls_seg(out)

        if self.training:
            return out, c_raw
        else:
            return out

    def loss(
        self,
        inputs: Tuple[Tensor],
        batch_data_samples: SampleList,
        train_cfg: ConfigType,
    ) -> Tuple[dict]:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            Tuple[dict[str, Tensor]]: a tuple of dictionary of loss components
        """
        seg_logits, c_raw = self.forward(inputs)
        aux_losses = self.aux_loss_by_paca(c_raw, batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses, aux_losses

    def aux_loss_by_paca(
        self, seg_logits: Tensor, batch_data_samples: SampleList
    ) -> dict:
        """Compute segmentation loss."""
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.aux_loss_decode, nn.ModuleList):
            aux_losses_decode = [self.aux_loss_decode]
        else:
            aux_losses_decode = self.aux_loss_decode
        for loss_decode in aux_losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )

        loss["acc_seg"] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index
        )
        return loss
