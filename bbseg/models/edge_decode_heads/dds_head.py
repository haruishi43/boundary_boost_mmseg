#!/usr/bin/env python3

"""DDS (Deep Diverse Supervision)

https://arxiv.org/pdf/1804.02864.pdf
"""

import torch.nn as nn

from mmseg.registry import MODELS

from .base_edge_decode_head import BaseEdgeDecodeHead
from ..utils import BasicBlockSideConv, GroupedConvFuse


@MODELS.register_module()
class DDSHead(BaseEdgeDecodeHead):
    def __init__(
        self,
        pred_key="fuse",
        log_keys=("fuse", "side5", "side4"),
        loss_decode=dict(
            binary=dict(
                side1=dict(type="BinaryEdgeLoss", loss_weight=1.0),
                side2=dict(type="BinaryEdgeLoss", loss_weight=1.0),
                side3=dict(type="BinaryEdgeLoss", loss_weight=1.0),
                side4=dict(type="BinaryEdgeLoss", loss_weight=1.0),
            ),
            mlbl=dict(
                side5=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
                fuse=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
            ),
        ),
        num_blocks=2,
        dilations=None,
        resize_index=-1,
        **kwargs,
    ) -> None:
        super().__init__(
            input_transform="multiple_select",
            pred_key=pred_key,
            log_keys=log_keys,
            loss_decode=loss_decode,
            **kwargs,
        )

        self.resize_index = resize_index

        _interp = "bilinear"  # nearest
        _bias = True
        _last_bias = True
        _fuse_bias = True

        # bias should not be turn on when some of the sides are not supervised

        sides = []
        for i in range(len(self.in_channels) - 1):
            sides.append(
                BasicBlockSideConv(
                    in_channels=self.in_channels[i],
                    out_channels=1,
                    num_blocks=num_blocks,
                    dilations=dilations,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=_bias,
                    act_cfg=self.act_cfg,
                    interpolation=_interp,
                    align_corners=self.align_corners,
                )
            )

        # last side is semantic
        sides.append(
            BasicBlockSideConv(
                in_channels=self.in_channels[-1],
                out_channels=self.num_classes,
                num_blocks=num_blocks,
                dilations=dilations,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                bias=_last_bias,
                act_cfg=self.act_cfg,
                interpolation=_interp,
                align_corners=self.align_corners,
            )
        )

        self.sides = nn.ModuleList(sides)
        self.fuse = GroupedConvFuse(
            num_classes=self.num_classes,
            num_sides=len(sides),
            conv_cfg=self.conv_cfg,
            bias=_fuse_bias,
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        if self.pass_input_image:
            h, w = x[self.resize_index].shape[2:]
            _ = x.pop()  # remove image if exists
            resize_to = (h, w)
        else:
            h, w = x[self.resize_index].shape[2:]
            resize_to = (h, w)

        side_outs = []
        for i, layer in enumerate(self.sides):
            side_outs.append(layer(x[i], resize_to))

        fuse = self.fuse(side_outs)

        outs = dict(fuse=fuse)
        for i, side_out in enumerate(side_outs):
            outs[f"side{i + 1}"] = side_out

        return outs
