#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


def weighted_binary_loss(
    edge: torch.Tensor,
    edge_label: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    reduction: str = "mean",
    ignore_index: int = 255,
) -> torch.Tensor:
    """Weighted Binary Binary Cross Entropy Loss."""
    # input edge dim=4 (b, 1, h, w)
    # input edge_label dim=4 (b, 1, h, w)
    pos_index = edge_label == 1
    neg_index = edge_label == 0
    ignore_index = edge_label == ignore_index

    # just set ignore_index to 0 to obtain loss
    edge_label[ignore_index] = 0

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
    edge: torch.Tensor,
    edge_label: torch.Tensor,
    sensitivity: int = 10,
    reduction: str = "mean",
    ignore_index: int = 255,
) -> torch.Tensor:
    # input edge dim=4 (b, 1, h, w)
    # input edge_label dim=4 (b, 1, h, w)
    ignore_index = edge_label == ignore_index
    pos_index = edge_label == 1
    neg_index = edge_label == 0

    # just set ignore_index to 0 to obtain loss
    edge_label[ignore_index] = 0

    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    pos_weight = (neg_num / pos_num).clamp(min=1, max=sum_num) / sensitivity

    w = torch.tensor([pos_weight], device=edge.device)

    loss = F.binary_cross_entropy_with_logits(
        edge,
        edge_label.float(),
        reduction="none",
        pos_weight=w.reshape(1, 1, 1, 1),
    )

    loss = loss * (1 - ignore_index.float())

    if reduction == "mean":
        return loss.mean()
    else:
        return loss


@MODELS.register_module()
class BinaryEdgeLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        loss_name: str = "loss_binary_edge",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self._alpha = alpha
        self._beta = beta

    def forward(
        self,
        edge: torch.Tensor,  # logits
        edge_label: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        **kwargs,
    ) -> torch.Tensor:
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


@MODELS.register_module()
class ConsensusBinaryEdgeLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        loss_name: str = "loss_conbin_edge",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        edge: torch.Tensor,  # logits
        edge_label: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        **kwargs,
    ) -> torch.Tensor:
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
