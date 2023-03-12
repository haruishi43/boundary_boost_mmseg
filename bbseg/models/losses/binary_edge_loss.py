#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def weighted_binary_loss(
    edge,
    edge_label,
    alpha=1.0,
    beta=1.0,
    reduction="mean",
    ignore_index=255,
):
    # input edge dim=4 (b, 1, h, w)
    # input edge_label dim=4 (b, 1, h, w)
    pos_index = edge_label == 1
    neg_index = edge_label == 0
    ignore_index = edge_label > 1  # should be `ignore_index`

    weight = torch.Tensor(edge.size()).fill_(0)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = alpha * neg_num / sum_num
    weight[neg_index] = beta * pos_num / sum_num
    weight[ignore_index] = 0
    weight = weight.to(edge.device)

    return F.binary_cross_entropy_with_logits(
        edge,
        edge_label.float(),
        weight,
        reduction=reduction,
    )


def balanced_binary_loss(
    edge,
    edge_label,
    reduction="mean",
    ignore_index=255,
    sensitivity=10,
):
    # input edge dim=4 (b, 1, h, w)
    # input edge_label dim=4 (b, 1, h, w)
    pos_index = edge_label == 1
    neg_index = edge_label == 0

    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    pos_weight = (neg_num / pos_num).clamp(min=1, max=sum_num) / sensitivity

    w = torch.tensor([pos_weight], device=edge.device)

    return F.binary_cross_entropy_with_logits(
        edge,
        edge_label.float(),
        reduction=reduction,
        pos_weight=w.reshape(1, 1, 1, 1),
    )


@LOSSES.register_module()
class BinaryEdgeLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        alpha=1.0,
        beta=1.0,
        loss_name="loss_binary_edge",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self._alpha = alpha
        self._beta = beta

    def forward(
        self,
        edge,  # logits
        edge_label,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        return self.loss_weight * weighted_binary_loss(
            edge=edge,
            edge_label=edge_label,
            alpha=self._alpha,
            beta=self._beta,
            reduction="mean",
            ignore_index=ignore_index,
        )

    @property
    def loss_name(self):
        return self._loss_name


@LOSSES.register_module()
class ConsensusBinaryEdgeLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        loss_name="loss_conbin_edge",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        edge,  # logits
        edge_label,
        weight=None,
        ignore_index=255,
        **kwargs,
    ):
        return self.loss_weight * weighted_binary_loss(
            edge=edge,
            edge_label=edge_label,
            alpha=1.0,
            beta=1.1,
            reduction="mean",
            ignore_index=ignore_index,
        )

    @property
    def loss_name(self):
        return self._loss_name
