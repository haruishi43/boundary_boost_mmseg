#!/usr/bin/env python3

"""Implementation of DFF

https://github.com/Lavender105/DFF/blob/master/exps/models/dff.py
"""

import torch.nn as nn

from .base_multisupervision_head import BaseMultiSupervisionHead
from ..builder import HEADS
from ..utils import (
    GeneralizedLocationAdaptiveLearner,
    SideConv,
)


@HEADS.register_module()
class GeneralizedDFFHead(BaseMultiSupervisionHead):
    def __init__(
        self,
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

        _interp = "bilinear"  # nearest
        _bias = True

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

        self.side_w = SideConv(
            in_channels=self.in_channels[-1],
            out_channels=self.num_classes * 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=_bias,  # might not need?
            act_cfg=self.act_cfg,
            interpolation=_interp,
            align_corners=self.align_corners,
        )

        self.ada_learner = GeneralizedLocationAdaptiveLearner(
            num_sides=len(sides),
            in_channels=self.num_classes * 4,
            out_channels=self.num_classes * 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs):
        x = [i for i in inputs]
        assert isinstance(x, list)
        # get the input image size
        bs, c, h, w = x[-1].shape
        resize_to = (h, w)  # TODO: might be too large

        # remove the input image and unused features
        x = [x[i] for i in self.in_index]

        side_outs = []
        for i, layer in enumerate(self.sides):
            side_outs.append(layer(x[i], resize_to))

        last = side_outs[-1]

        side_outs.append(self.side_w(x[-1], resize_to))
        fuse = self.ada_learner(side_outs)

        return dict(fuse=fuse, last=last)


@HEADS.register_module()
class AuxDFFHead(GeneralizedDFFHead):
    ...
