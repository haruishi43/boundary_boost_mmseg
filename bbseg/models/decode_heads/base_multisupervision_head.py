#!/usr/bin/env python3

"""Deep Supervision Head

Supports binary and multilabel edges for supervision.
"""

import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize

from ..builder import build_loss
from ..losses import calc_metrics
from .utils import init_loss


class BaseMultiSupervisionHead(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels,
        channels,
        *,
        num_classes,
        edge_key,
        log_edge_keys,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        in_index=-1,
        input_transform=None,
        binary_keys=[],
        multilabel_keys=[],
        binary_loss_weights=[],
        multilabel_loss_weights=[],
        loss_binary=None,
        loss_multilabel=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
        ignore_index=255,
        sampler=None,
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01),
        no_accuracy=False,  # TODO: this is a hack to avoid memory errors
    ):
        super().__init__(init_cfg)

        assert isinstance(edge_key, str)
        self.edge_key = edge_key
        if log_edge_keys is None:
            log_edge_keys = []
        elif isinstance(log_edge_keys, str):
            log_edge_keys = [log_edge_keys]
        assert isinstance(log_edge_keys, (tuple, list))
        self.log_edge_keys = log_edge_keys

        self.loss_multilabel = init_loss(
            loss_multilabel, build_loss, num_classes=num_classes
        )
        if self.loss_multilabel is None:
            if len(multilabel_keys) > 0:
                warnings.warn(
                    "multilabel loss is None, but there seems to be some keys, "
                    f"{multilabel_keys},"
                    "removing keys..."
                )
                multilabel_keys = []
        else:
            if len(multilabel_loss_weights) == 0:
                multilabel_loss_weights = [1.0 for _ in multilabel_keys]

        self.loss_binary = init_loss(loss_binary, build_loss)
        if self.loss_binary is None:
            assert (
                self.loss_multilabel is not None
            ), "needs either binary or multilabel loss"
            if len(binary_keys) > 0:
                warnings.warn(
                    "binary loss is None, but there seems to be some keys, "
                    f"{binary_keys},"
                    "removing keys..."
                )
                binary_keys = []
        else:
            if len(binary_loss_weights) == 0:
                binary_loss_weights = [1.0 for _ in binary_keys]

        assert len(multilabel_keys) == len(multilabel_loss_weights)
        assert len(binary_keys) == len(binary_loss_weights)
        assert (
            len(multilabel_keys) + len(binary_keys) > 0
        ), f"there seems to be no edges to supervise: {multilabel_keys}, {binary_keys}"

        # loss info (key: weight)
        self.multilabel_info = dict(zip(multilabel_keys, multilabel_loss_weights))
        self.binary_info = dict(zip(binary_keys, binary_loss_weights))

        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.fp16_enabled = False

        self.no_accuracy = no_accuracy

    def extra_repr(self):
        """Extra repr."""
        s = (
            f"input_transform={self.input_transform}, "
            f"ignore_index={self.ignore_index}, "
            f"align_corners={self.align_corners}"
        )
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs, **kwargs):
        """Placeholder of forward function."""
        pass

    def forward_train(
        self,
        inputs,
        img_metas,
        gt_semantic_edge,
        train_cfg,
    ):
        logits = self(inputs)

        losses = dict()
        if self.multilabel_losses is not None:
            losses.update(self.multilabel_losses(logits, gt_semantic_edge))
        if self.binary_losses is not None:
            losses.update(self.binary_losses(logits, gt_semantic_edge))

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """use test_cfg's edge_key to choose which output"""
        edge_key = test_cfg.get("edge_key", self.edge_key)
        return self(inputs)[edge_key]

    @force_fp32(apply_to=("logits"))
    def binary_losses(self, logits, label):
        """Compute binary edge loss."""

        if isinstance(logits, torch.Tensor):
            # if a tensor is passed
            logits = dict(binary=logits)
        assert isinstance(logits, dict)

        loss = dict()

        # convert multilabel to binary edge, if needed
        if label.shape[1] != 1:
            # convert to binary
            label = (torch.sum(label, axis=1) > 0).unsqueeze(1).float()

        for k, logit in logits.items():
            if k in self.binary_info.keys():

                # get key specific weight
                w = self.binary_info[k]

                logit = resize(
                    input=logit,
                    size=label.shape[2:],  # (b, cls, h, w)
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                assert (
                    label.shape == logit.shape
                ), f"label, pred: {label.shape}, {logit.shape}"

                if not isinstance(self.loss_binary, nn.ModuleList):
                    losses_edge = [self.loss_binary]
                else:
                    losses_edge = self.loss_binary

                for loss_edge in losses_edge:
                    if loss_edge.loss_name not in loss:
                        loss[loss_edge.loss_name] = w * loss_edge(
                            logit,
                            label,
                            ignore_index=self.ignore_index,
                        )
                    else:
                        loss[loss_edge.loss_name] += w * loss_edge(
                            logit,
                            label,
                            ignore_index=self.ignore_index,
                        )

                if (k in self.log_edge_keys) and (not self.no_accuracy):
                    for name, v in calc_metrics(logit, label).items():
                        loss[k + "_" + name] = v

        return loss

    @force_fp32(apply_to=("logits"))
    def multilabel_losses(self, logits, label):
        """Compute multilabel edge loss."""

        if isinstance(logits, torch.Tensor):
            # if a tensor is passed
            logits = dict(multilabel=logits)
        assert isinstance(logits, dict)

        loss = dict()

        for k, logit in logits.items():
            if k in self.multilabel_info.keys():
                # get key specific weight
                w = self.multilabel_info[k]

                logit = resize(
                    input=logit,
                    size=label.shape[2:],  # (b, cls, h, w)
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                assert (
                    label.shape == logit.shape
                ), f"label, pred: {label.shape}, {logit.shape}"

                if not isinstance(self.loss_multilabel, nn.ModuleList):
                    losses_edge = [self.loss_multilabel]
                else:
                    losses_edge = self.loss_multilabel

                for loss_edge in losses_edge:
                    if loss_edge.loss_name not in loss:
                        loss[loss_edge.loss_name] = w * loss_edge(
                            logit,
                            label,
                            ignore_index=self.ignore_index,
                        )
                    else:
                        loss[loss_edge.loss_name] += w * loss_edge(
                            logit,
                            label,
                            ignore_index=self.ignore_index,
                        )

                if (k in self.log_edge_keys) and (not self.no_accuracy):
                    for name, v in calc_metrics(logit, label).items():
                        loss[k + "_" + name] = v

        return loss
