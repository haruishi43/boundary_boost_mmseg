#!/usr/bin/env python3

"""Implementation of DFF

https://github.com/Lavender105/DFF/blob/master/exps/models/dff.py
"""

import torch.nn as nn

from mmseg.registry import MODELS

from .base_edge_decode_head import BaseEdgeDecodeHead
from ..utils import (
    GeneralizedLocationAdaptiveLearner,
    SideConv,
)


@MODELS.register_module()
class DFFHead(BaseEdgeDecodeHead):
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
        use_pre_act=True,
        use_sigmoid=False,
        **kwargs,
    ) -> None:
        """DFF Head for various backbones.

        In the original implementation,
        - `use_pre_act` is set to False
        - `use_sigmoid` is set to False
        """
        super().__init__(
            input_transform="multiple_select",
            pred_key=pred_key,
            log_keys=log_keys,
            loss_decode=loss_decode,
            **kwargs,
        )

        self.resize_index = resize_index

        _interp = "bilinear"  # nearest
        _weight_bias = False
        _side_bias = False
        _last_bias = True

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

        self.side_w = SideConv(
            in_channels=self.in_channels[-1],
            out_channels=self.num_classes * 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=_weight_bias,
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
            use_pre_act=use_pre_act,
            use_sigmoid=use_sigmoid,
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

        last = side_outs[-1]

        side_outs.append(self.side_w(x[-1], resize_to))
        fuse = self.ada_learner(side_outs)

        return dict(fuse=fuse, last=last)
