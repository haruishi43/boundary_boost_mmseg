#!/usr/bin/env python3

"""Implementation of CASENet head."""

import torch.nn as nn

from mmseg.registry import MODELS

from .base_edge_decode_head import BaseEdgeDecodeHead
from ..utils import GroupedConvFuse, SideConv


@MODELS.register_module()
class CASENetHead(BaseEdgeDecodeHead):
    def __init__(
        self,
        pred_key="fuse",
        log_keys=("fuse", "last"),
        loss_decode=dict(
            mlbl=dict(
                fuse=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
                last=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
            ),
        ),
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
        _side_bias = False
        _last_bias = True
        _fuse_bias = True

        # bias should not be turn on when some of the sides are not supervised

        sides = []
        for i in range(len(self.in_channels) - 1):
            sides.append(
                SideConv(
                    in_channels=self.in_channels[i],
                    out_channels=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=_side_bias,
                    act_cfg=self.act_cfg,
                    interpolation=_interp,
                    align_corners=self.align_corners,
                )
            )

        # last side is semantic
        sides.append(
            SideConv(
                in_channels=self.in_channels[-1],
                out_channels=self.num_classes,
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

        return dict(fuse=fuse, last=side_outs[-1])
