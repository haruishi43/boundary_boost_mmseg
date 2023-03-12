#!/usr/bin/env python3

"""DDS (Deep Diverse Supervision)

https://arxiv.org/pdf/1804.02864.pdf
"""

import torch.nn as nn

from .base_multisupervision_head import BaseMultiSupervisionHead
from ..builder import HEADS
from ..utils import BasicBlockSideConv, GroupedConvFuse


@HEADS.register_module()
class GeneralizedDDSHead(BaseMultiSupervisionHead):
    def __init__(
        self,
        num_blocks=2,
        dilations=None,
        side_resize_index=-1,
        edge_key="fuse",
        log_edge_keys=("fuse", "side5", "side4"),
        binary_keys=("side1", "side2", "side3", "side4"),
        multilabel_keys=("side5", "fuse"),
        loss_binary=dict(type="BinaryEdgeLoss", loss_weight=1.0),
        loss_multilabel=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
        **kwargs,
    ):
        super().__init__(
            input_transform="multiple_select",
            edge_key=edge_key,
            log_edge_keys=log_edge_keys,
            binary_keys=binary_keys,
            multilabel_keys=multilabel_keys,
            loss_binary=loss_binary,
            loss_multilabel=loss_multilabel,
            **kwargs,
        )

        self.side_resize_index = side_resize_index

        _interp = "bilinear"  # nearest
        _bias = True

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
                bias=_bias,
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
            bias=_bias,  # originally true
        )

    def forward(self, inputs):
        x = [i for i in inputs]
        assert isinstance(x, list)
        # get the input image size
        bs, c, h, w = x[self.side_resize_index].shape
        resize_to = (h, w)  # TODO: might be too large

        # remove the input image and unused features
        x = [x[i] for i in self.in_index]

        side_outs = []
        for i, layer in enumerate(self.sides):
            side_outs.append(layer(x[i], resize_to))

        fuse = self.fuse(side_outs)

        outs = dict(fuse=fuse)
        for i, side_out in enumerate(side_outs):
            outs[f"side{i + 1}"] = side_out

        return outs


@HEADS.register_module()
class AuxDDSHead(GeneralizedDDSHead):
    ...
