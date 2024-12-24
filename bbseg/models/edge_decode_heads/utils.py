#!/usr/bin/env python3

from typing import Dict

import torch.nn as nn

from mmseg.registry import MODELS


def init_edge_loss(loss_decode: Dict[str, dict], num_classes: int):
    """Initialize edge loss.

    Structure of `loss_decode`:

    .. code:: python
    loss_decode = dict(
        binary=dict(
            logit_name1=dict(
                type="CrossEntropyLoss",
                loss_weight=1.0,
            ),
            ...
        ),
        mlbl=dict(
            logit_name2=dict(
                type="MultiLabelEdgeLoss",
                loss_weight=1.0,
            ),
            logti_name3=[  # could be a list/tuple
                dict(
                    type="MultiLabelEdgeLoss",
                    loss_weight=1.0,
                ),
                ...
            ]
            ...
        ),
    )
    """

    # need to do some checks
    assert (
        isinstance(num_classes, int) and num_classes > 0
    ), f"num_classes must be a positive int, but got {num_classes}"

    binary_loss = loss_decode.get("binary", None)
    binary_keys = []
    mlbl_loss = loss_decode.get("mlbl", None)
    mlbl_keys = []

    assert (
        binary_loss is not None or mlbl_loss is not None
    ), "loss_decode must contain at least one of 'binary' or 'mlbl'"

    # binary
    if binary_loss is not None:
        assert isinstance(
            binary_loss, dict
        ), f"binary_loss must be a dict, but got {type(binary_loss)}"

        for k, losses in binary_loss.items():
            assert isinstance(
                k, str
            ), f"binary_loss key must be a str, but got {type(k)}"
            binary_keys.append(k)

            # we assume that losses are `num_classes=1` by default
            if isinstance(losses, dict):
                binary_loss[k] = MODELS.build(losses)
            elif isinstance(losses, (list, tuple)):
                binary_loss[k] = nn.ModuleList()
                for loss in losses:
                    binary_loss[k].append(MODELS.build(loss))
            else:
                raise TypeError(
                    f"binary loss for {k} must be a dict or sequence of dict,\
                    but got {type(losses)}"
                )

    # mulit-label
    if mlbl_loss is not None:
        assert isinstance(
            mlbl_loss, dict
        ), f"mlbl_loss must be a dict, but got {type(mlbl_loss)}"
        assert (
            num_classes > 1
        ), f"num_classes must be greater than 1 for mlbl_loss, but got {num_classes}"

        for k, losses in mlbl_loss.items():
            assert isinstance(k, str), f"mlbl_loss key must be a str, but got {type(k)}"
            mlbl_keys.append(k)

            # we assume that losses are `num_classes` by default
            if isinstance(losses, dict):
                if "num_classes" in losses.keys():
                    losses.update(dict(num_classes=num_classes))
                mlbl_loss[k] = MODELS.build(losses)
            elif isinstance(losses, (list, tuple)):
                mlbl_loss[k] = nn.ModuleList()
                for loss in losses:
                    if "num_classes" in loss.keys():
                        loss.update(dict(num_classes=num_classes))
                    mlbl_loss[k].append(MODELS.build(loss))
            else:
                raise TypeError(
                    f"loss {k} must be a dict or sequence of dict or None,\
                    but got {type(losses)}"
                )

    return binary_loss, binary_keys, mlbl_loss, mlbl_keys


def get_loss_names(losses):
    loss_names = []

    if not isinstance(losses, (list, tuple)):
        losses = [losses]

    for loss in losses:
        if loss is not None:
            if not isinstance(loss, nn.ModuleList):
                loss_names.append(loss.loss_name)
            else:
                for l in loss:
                    loss_names.append(l.loss_name)

    return loss_names
