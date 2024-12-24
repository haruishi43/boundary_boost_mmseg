#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn as nn
from mmseg.registry import MODELS

from .binary_edge_loss import balanced_binary_loss, weighted_binary_loss


@MODELS.register_module()
class MultiLabelEdgeLoss(nn.Module):
    """Class Balanced Multilabel Loss used in DFF"""

    def __init__(
        self,
        loss_weight: float = 1.0,
        loss_name: str = "loss_mlbl_edge",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        edge: torch.Tensor,  # logits
        edge_label: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss_total = 0

        # FIXME: could optimize for batched loss
        for i in range(edge_label.size(0)):  # iterate for batch size
            pred = edge[i]
            target = edge_label[i]

            num_pos = torch.sum(target)  # true positive number
            num_total = target.size(-1) * target.size(-2)  # true total number
            num_neg = num_total - num_pos
            # compute a pos_weight for each image
            pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total)

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = (
                pred
                - pred * target
                + log_weight
                * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())
            )

            loss = loss.mean()
            loss_total = loss_total + loss

        loss_total = loss_total / edge_label.size(0)
        return self.loss_weight * loss_total

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class BalancedMultiLabelLoss(nn.Module):
    def __init__(
        self,
        sensitivity: int = 10,
        loss_weight: float = 1.0,
        loss_name: str = "loss_balanced_mlbl_edge",
    ) -> None:
        super().__init__()
        self.sensitivity = sensitivity
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
        loss_total = 0

        for i in range(edge_label.size(1)):  # iterate for classes
            pred = edge[:, i, ...].unsqueeze(1)
            target = edge_label[:, i, ...].unsqueeze(1)

            loss = balanced_binary_loss(
                edge=pred,
                edge_label=target,
                sensitivity=self.sensitivity,
                ignore_index=ignore_index,
                reduction="mean",
            )
            loss_total = loss_total + loss

        return self.loss_weight * loss_total

    @property
    def loss_name(self):
        return self._loss_name


@MODELS.register_module()
class WeightedMultiLabelLoss(nn.Module):
    """Weighted Multi-label Loss

    Weighing `alpha=beta` results in high recall,
    so reducing `beta` by `1/num_classes` stabilizes.
    """

    def __init__(
        self,
        num_classes: int = 19,
        loss_weight: float = 1.0,
        loss_name: str = "loss_multilabel_bce_edge",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self._alpha = 1.0
        self._beta = float(num_classes)

    def forward(
        self,
        edge: torch.Tensor,  # logits
        edge_label: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        **kwargs,
    ) -> torch.Tensor:
        loss_total = 0

        for i in range(edge_label.size(1)):  # iterate for classes
            pred = edge[:, i, ...].unsqueeze(1)
            target = edge_label[:, i, ...].unsqueeze(1)

            loss = weighted_binary_loss(
                edge=pred,
                edge_label=target,
                alpha=self._alpha,
                beta=self._beta,
                ignore_index=ignore_index,
                reduction="mean",
            )
            loss_total = loss_total + loss

        return self.loss_weight * loss_total

    @property
    def loss_name(self):
        return self._loss_name
