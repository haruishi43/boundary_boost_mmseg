#!/usr/bin/env python

import warnings

import torch
import torch.nn as nn

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.models.backbones.resnet import BasicBlock, Bottleneck
from mmseg.models.backbones.hrnet import HRModule
from mmseg.ops import Upsample, resize

from ..builder import BACKBONES


class ModHRModule(HRModule):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(
        self,
        down_strides=(2, 2, 2),
        **kwargs,
    ):
        self.down_strides = down_strides
        super().__init__(**kwargs)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # we need to upsample j to i
                    # need to calculate the upsampling scale

                    scale = 1
                    for k in range(i, j):
                        scale *= self.down_strides[k]

                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            # we set align_corners=False for HRNet
                            Upsample(
                                scale_factor=scale,
                                mode="bilinear",
                                align_corners=False,
                            ),
                        )
                    )
                elif j == i:
                    # same resolution
                    fuse_layer.append(None)
                else:
                    # we need to downsample j to i
                    # employ multiple conv layers with strides (2 to downsample x2)
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            # final
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=self.down_strides[j + k],
                                        padding=1,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=self.down_strides[j + k],
                                        padding=1,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)


@BACKBONES.register_module()
class ModHRNet(BaseModule):
    """Modified HRNet backbone."""

    blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}

    def __init__(
        self,
        extra,
        in_channels=3,
        stem_channels=64,
        strides=(2, 2, 2, 2, 2),
        return_stem=False,
        return_sides=False,
        align_corners=False,
        interpolation="bilinear",
        resize_concat_sides=True,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        frozen_stages=-1,
        zero_init_residual=False,
        multiscale_output=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ModHRNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
        else:
            raise TypeError("pretrained must be a str or None")

        # Assert configurations of 4 stages are in extra
        assert (
            "stage1" in extra
            and "stage2" in extra
            and "stage3" in extra
            and "stage4" in extra
        )
        # Assert whether the length of `num_blocks` and `num_channels` are
        # equal to `num_branches`
        for i in range(4):
            cfg = extra[f"stage{i + 1}"]
            assert (
                len(cfg["num_blocks"]) == cfg["num_branches"]
                and len(cfg["num_channels"]) == cfg["num_branches"]
            )

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages

        self.return_stem = return_stem
        self.return_sides = return_sides
        self.interp = interpolation
        self.align_corners = align_corners
        if isinstance(resize_concat_sides, bool):
            self.resize_concat_sides = [resize_concat_sides] * 4
        else:
            self.resize_concat_sides = resize_concat_sides
        assert len(self.resize_concat_sides) == 4

        # stem net
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1
        )
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=2
        )
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=3,
            stride=strides.pop(0),
            padding=1,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            stem_channels,
            stem_channels,
            kernel_size=3,
            stride=strides.pop(0),  # HACK: pop the first one
            padding=1,
            bias=False,
        )
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        block_type = self.stage1_cfg["block"]
        num_blocks = self.stage1_cfg["num_blocks"][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, stem_channels, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block_type = self.stage2_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channels],
            num_channels,
            strides[0],
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg,
            num_channels,
            down_strides=strides,
        )

        # stage 3
        self.stage3_cfg = self.extra["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block_type = self.stage3_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
            strides[1],
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            down_strides=strides,
        )

        # stage 4
        self.stage4_cfg = self.extra["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block_type = self.stage4_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
            strides[2],
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=multiscale_output,
            down_strides=strides,
        )

        self._freeze_stages()

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(
        self,
        num_channels_pre_layer,
        num_channels_cur_layer,
        down_stride=2,
    ):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[
                                1
                            ],
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # Generates downsampled feature maps
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=down_stride,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make each layer."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        block_init_cfg = None
        if (
            self.pretrained is None
            and not hasattr(self, "init_cfg")
            and self.zero_init_residual
        ):
            if block is BasicBlock:
                block_init_cfg = dict(
                    type="Constant", val=0, override=dict(name="norm2")
                )
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type="Constant", val=0, override=dict(name="norm3")
                )

        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=block_init_cfg,
            )
        )
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=block_init_cfg,
                )
            )

        return Sequential(*layers)

    def _make_stage(
        self,
        layer_config,
        in_channels,
        multiscale_output=True,
        down_strides=(2, 2, 2),
    ):
        """Make each stage."""
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = self.blocks_dict[layer_config["block"]]

        hr_modules = []
        block_init_cfg = None
        if (
            self.pretrained is None
            and not hasattr(self, "init_cfg")
            and self.zero_init_residual
        ):
            if block is BasicBlock:
                block_init_cfg = dict(
                    type="Constant", val=0, override=dict(name="norm2")
                )
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type="Constant", val=0, override=dict(name="norm3")
                )

        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                ModHRModule(
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    in_channels=in_channels,
                    num_channels=num_channels,
                    multiscale_output=reset_multiscale_output,
                    down_strides=down_strides,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    block_init_cfg=block_init_cfg,
                )
            )

        return Sequential(*hr_modules), in_channels

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:

            self.norm1.eval()
            self.norm2.eval()
            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                m = getattr(self, f"layer{i}")
                t = getattr(self, f"transition{i}")
            elif i == 4:
                m = getattr(self, f"stage{i}")
            else:
                m = getattr(self, f"stage{i}")
                t = getattr(self, f"transition{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            t.eval()
            for param in t.parameters():
                param.requires_grad = False

    def _resize_and_concat(self, xs):
        xs = [
            resize(
                x,
                size=xs[0].shape[2:],
                mode=self.interp,
                align_corners=self.align_corners,
            )
            for x in xs
        ]
        x = torch.cat(xs, dim=1)
        return x

    def forward(self, x):
        """Forward function."""

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        outs = []
        if self.return_stem:
            # add stem
            outs.append(x)

        # stage 1
        x = self.layer1(x)
        if self.return_sides:
            outs.append(x)

        # stage 2
        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        if self.return_sides:
            if self.resize_concat_sides[1]:
                outs.append(self._resize_and_concat(y_list))
            else:
                outs.append(y_list[0])

        # stage 3
        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        if self.return_sides:
            if self.resize_concat_sides[2]:
                outs.append(self._resize_and_concat(y_list))
            else:
                outs.append(y_list[0])

        # stage 4
        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        if self.return_sides:
            if self.resize_concat_sides[3]:
                outs.append(self._resize_and_concat(y_list))
            else:
                # might be redundant
                outs.append(y_list[0])

        outs += y_list
        return outs

    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
