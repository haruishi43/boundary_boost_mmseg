#!/usr/bin/env python3

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.decode_heads.aspp_head import ASPPHead
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPModule
from mmseg.ops import resize

from ..builder import HEADS


@HEADS.register_module()
class ModDepthwiseSeparableASPPHead(ASPPHead):
    """Modded DeepLabV3+ Head

    Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.
    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
        c1_in_index (int): index of c1 feature
    """

    def __init__(self, c1_in_channels, c1_channels, c1_in_index=0, **kwargs):
        super(ModDepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0

        self.c1_in_index = c1_in_index

        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[self.c1_in_index])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output
