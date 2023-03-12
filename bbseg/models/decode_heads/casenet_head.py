#!/usr/bin/env python3

import torch.nn as nn

from .base_multisupervision_head import BaseMultiSupervisionHead
from ..builder import HEADS
from ..utils import GroupedConvFuse, SideConv


@HEADS.register_module()
class GeneralizedCASENetHead(BaseMultiSupervisionHead):
    def __init__(
        self,
        resize_index=-1,  # input image size
        edge_key="fuse",
        log_edge_keys=("fuse", "last"),
        binary_keys=[],
        multilabel_keys=("fuse", "last"),
        loss_binary=None,
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

        self.resize_index = resize_index

        _interp = "bilinear"  # nearest
        _bias = False

        # bias should not be turn on when some of the sides are not supervised

        sides = []

        for i in range(len(self.in_channels) - 1):
            sides.append(
                SideConv(
                    in_channels=self.in_channels[i],
                    out_channels=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,  # bias instead of bn
                    bias=_bias,  # add bias in the last layer
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
                norm_cfg=self.norm_cfg,  # bias instead of bn
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
        bs, c, h, w = x[self.resize_index].shape
        resize_to = (h, w)  # TODO: might be too large

        # remove the input image and unused features
        x = [x[i] for i in self.in_index]

        side_outs = []
        for i, layer in enumerate(self.sides):
            side_outs.append(layer(x[i], resize_to))

        fuse = self.fuse(side_outs)

        return dict(fuse=fuse, last=side_outs[-1])


@HEADS.register_module()
class AuxCASENetHead(GeneralizedCASENetHead):
    ...
