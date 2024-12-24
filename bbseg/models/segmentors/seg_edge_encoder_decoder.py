#!/usr/bin/env python3

"""SegEdgeEncoderDecoder.

EncoderDecoder with edge branch.

TODO: revise docstrings.
"""

import logging
from typing import List

import torch.nn.functional as F
from torch import Tensor

from mmengine.logging import print_log
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.utils import (
    ConfigType,
    OptConfigType,
    OptSampleList,
    SampleList,
    add_prefix,
)

from bbseg.structures import SegEdgeDataSample


@MODELS.register_module()
class SegEdgeEncoderDecoder(EncoderDecoder):
    """EncoderDecoder Segmentor with Auxiliary Edge Modules."""

    def __init__(
        self,
        edge_decode_head: ConfigType,
        edge_neck: OptConfigType = None,
        pass_input_image: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Initialize edge modules
        self._init_edge_decode_head(edge_decode_head)
        if edge_neck is not None:
            self.edge_neck = MODELS.build(edge_neck)

        self.pass_input_image = pass_input_image

    @property
    def with_edge_neck(self) -> bool:
        """bool: whether the segmentor has edge_neck"""
        return hasattr(self, "edge_neck") and self.edge_neck is not None

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""

        x = self.backbone(inputs)
        if self.with_neck:
            seg_x = self.neck(x)
        else:
            seg_x = x

        if self.with_edge_neck:
            edge_x = self.edge_neck(x)
        else:
            edge_x = x

        # FIXME: we might want to pass the input before the neck
        if self.pass_input_image:
            edge_x = (*edge_x, inputs)

        return seg_x, edge_x

    def _init_edge_decode_head(self, edge_decode_head: ConfigType) -> None:
        self.edge_decode_head = MODELS.build(edge_decode_head)
        self.num_edge_classes = self.edge_decode_head.num_classes

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        seg_x, edge_x = self.extract_feat(inputs)

        seg_logits = self.decode_head.predict(
            seg_x,
            batch_img_metas,
            self.test_cfg,
        )
        edge_logits = self.edge_decode_head.predict(
            edge_x,
            batch_img_metas,
            self.test_cfg,
        )
        return seg_logits, edge_logits

    def _edge_head_forward_train(
        self, inputs: List[Tensor], data_samples: SampleList
    ) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.edge_decode_head.loss(inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_aux, "edge"))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_x, edge_x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(seg_x, data_samples)
        losses.update(loss_decode)

        loss_edge = self._edge_head_forward_train(edge_x, data_samples)
        losses.update(loss_edge)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(seg_x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ] * inputs.shape[0]

        seg_logits, edge_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, edge_logits, data_samples)

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """

        x = self.backbone(inputs)
        if self.with_neck:
            seg_x = self.neck(x)
        else:
            seg_x = x

        if self.with_edge_neck:
            edge_x = self.edge_neck(x)
        else:
            edge_x = x

        seg_logits = self.decode_head.forward(seg_x)
        edge_logits = self.edge_decode_head.forward(edge_x)

        return seg_logits, edge_logits

    def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        edge_out_channels = self.num_edge_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        seg_preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        edge_preds = inputs.new_zeros((batch_size, edge_out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]["img_shape"] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit, crop_edge_logit = self.encode_decode(
                    crop_img,
                    batch_img_metas,
                )
                seg_preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(seg_preds.shape[3] - x2),
                        int(y1),
                        int(seg_preds.shape[2] - y2),
                    ),
                )
                edge_preds += F.pad(
                    crop_edge_logit,
                    (
                        int(x1),
                        int(edge_preds.shape[3] - x2),
                        int(y1),
                        int(edge_preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = seg_preds / count_mat
        edge_logits = edge_preds / count_mat

        return seg_logits, edge_logits

    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits, edge_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits, edge_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> List[Tensor]:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            List[Tensor]: segmentation and edge logits
        """
        assert self.test_cfg.get("mode", "whole") in ["slide", "whole"], (
            f'Only "slide" or "whole" test mode are supported, but got '
            f'{self.test_cfg["mode"]}.'
        )
        ori_shape = batch_img_metas[0]["ori_shape"]
        if not all(_["ori_shape"] == ori_shape for _ in batch_img_metas):
            print_log(
                "Image shapes are different in the batch.",
                logger="current",
                level=logging.WARN,
            )
        if self.test_cfg.mode == "slide":
            seg_logit, edge_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit, edge_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit, edge_logit

    def postprocess_result(
        self,
        seg_logits: Tensor,
        edge_logits: Tensor,
        data_samples: OptSampleList = None,
    ) -> SampleList:
        """Convert results list to `SegEdgeDataSample`.
        Args:
            bin_edge_logits (Tensor): The edge results, edge_logits from
                model of each input image.
            data_samples (list[:obj:`BasicEdgeDataSample`]): The edge data samples.
                It usually includes information such as `metainfo` and
                `gt_bin_edge`. Default to None.
        Returns:
            list[:obj:`BasicEdgeDataSample`]: Edge results of the
            input images. Each BasicEdgeDataSample usually contain:

            - ``pred_bin_edge``(PixelData): Prediction of binary edge.
            - ``bin_edge_logits``(PixelData): Predicted logits of binary
                edge before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegEdgeDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if "img_padding_size" not in img_meta:
                    padding_size = img_meta.get("padding_size", [0] * 4)
                else:
                    padding_size = img_meta["img_padding_size"]
                padding_left, padding_right, padding_top, padding_bottom = padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[
                    i : i + 1,
                    :,
                    padding_top : H - padding_bottom,
                    padding_left : W - padding_right,
                ]
                # i_bin_edge_logits shape is 1, C, H, W after remove padding
                i_edge_logits = edge_logits[
                    i : i + 1,
                    :,
                    padding_top : H - padding_bottom,
                    padding_left : W - padding_right,
                ]

                flip = img_meta.get("flip", None)
                if flip:
                    flip_direction = img_meta.get("flip_direction", None)
                    assert flip_direction in ["horizontal", "vertical"]
                    if flip_direction == "horizontal":
                        i_seg_logits = i_seg_logits.flip(dims=(3,))
                        i_edge_logits = i_edge_logits.flip(dims=(3,))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2,))
                        i_edge_logits = i_edge_logits.flip(dims=(2,))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta["ori_shape"],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                ).squeeze(0)
                i_edge_logits = resize(
                    i_edge_logits,
                    size=img_meta["ori_shape"],
                    mode="bilinear",
                    align_corners=self.align_corners,
                    warning=False,
                ).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]
                i_edge_logits = edge_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > self.decode_head.threshold).to(
                    i_seg_logits
                )

            data_samples[i].set_data(
                {
                    "seg_logits": PixelData(**{"data": i_seg_logits}),
                    "pred_sem_seg": PixelData(**{"data": i_seg_pred}),
                    "edge_logits": PixelData(**{"data": i_edge_logits}),
                    "pred_edge": PixelData(**{"data": i_edge_logits.sigmoid()}),
                }
            )

        return data_samples

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit, edge_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit, cur_edge_logit = self.inference(
                inputs[i], batch_img_metas[i], rescale
            )
            seg_logit += cur_seg_logit
            edge_logit += cur_edge_logit
        seg_logit /= len(inputs)
        edge_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        edge_pred = edge_logit.sigmoid() > self.edge_decode_head.threshold
        # unravel batch dim
        seg_pred = list(seg_pred)
        edge_pred = list(edge_pred)
        return seg_pred, edge_pred
