#!/usr/bin/env python3

"""Custom EncoderDecoder that subclasses the original version.

- Support methods that depends on passing input image to the head.
- Need to use mmseg dataset class.
- Only supports mmseg registered LOSSES (SEG_LOSSES).
"""

import torch.nn as nn

from mmseg.core import add_prefix
from mmseg.models.segmentors import EncoderDecoder

from .. import builder
from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    """Custom Encoder Decoder segmentors.

    - pass input image to decode head
    """

    def __init__(
        self,
        *args,
        pass_input_image=False,
        **kwargs,
    ):
        super(CustomEncoderDecoder, self).__init__(*args, **kwargs)

        # pass input image to decode head
        self.pass_input_image = pass_input_image

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        if self.pass_input_image:
            # HACK: appends image to the end of the inputs
            x = (*x, img)

        return x


@SEGMENTORS.register_module()
class SBCBEncoderDecoder(CustomEncoderDecoder):
    """SBCB Encoder Decoder Module.

    This module is only used to train the auxiliary model for SBCB.
    For inference, you can remove the auxiliary head using the config or
    load the weights into a normal EncoderDecoder model.
    """

    def __init__(self, *args, auxiliary_edge_head=None, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize auxiliary edge head
        self._init_auxiliary_edge_head(auxiliary_edge_head)

    @property
    def with_auxiliary_edge_head(self):
        """bool: whether the segmentor has auxiliary edge head for SBCB"""
        return (
            hasattr(self, "auxiliary_edge_head")
            and self.auxiliary_edge_head is not None
        )

    def _init_auxiliary_edge_head(self, auxiliary_edge_head):
        if auxiliary_edge_head is not None:
            if isinstance(auxiliary_edge_head, list):
                self.auxiliary_edge_head = nn.ModuleList()
                for head_cfg in auxiliary_edge_head:
                    self.auxiliary_edge_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_edge_head = builder.build_head(auxiliary_edge_head)

    def _auxiliary_edge_head_forward_train(self, x, img_metas, gt_semantic_edge):
        """Forward function for auxiliary edge head in training."""

        losses = dict()
        if isinstance(self.auxiliary_edge_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_edge_head):
                loss_aux = aux_head.forward_train(
                    x, img_metas, gt_semantic_edge, self.train_cfg
                )
                losses.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_aux = self.auxiliary_edge_head.forward_train(
                x, img_metas, gt_semantic_edge, self.train_cfg
            )
            losses.update(add_prefix(loss_aux, "aux"))

        return losses

    def forward_train(
        self, img, img_metas, gt_semantic_seg, gt_semantic_edge=None, **kwargs
    ):
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x,
            img_metas,
            gt_semantic_seg,
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x,
                img_metas,
                gt_semantic_seg,
            )
            losses.update(loss_aux)

        if self.with_auxiliary_edge_head:
            assert (
                gt_semantic_edge is not None
            ), "auxiliary edge exists but gt_semantic_edge is None"
            loss_aux_edge = self._auxiliary_edge_head_forward_train(
                x,
                img_metas,
                gt_semantic_edge,
            )
            losses.update(loss_aux_edge)

        return losses
