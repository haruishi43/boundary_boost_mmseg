#!/usr/bin/env python3

"""Extend and add tranformations.

Skipped:
- RandomCutOut
- RandomMosaic
"""

import numpy as np

import mmcv
from mmseg.datasets.pipelines.transforms import (
    Resize as MMSEG_Resize,
    Pad as MMSEG_Pad,
    RandomRotate as MMSEG_RandomRotate,
    RandomCrop as MMSEG_RandomCrop,
)

from ..builder import PIPELINES


@PIPELINES.register_module()
class AddIgnoreBorder(object):
    """Add ignore border to the semantic segmentation map.

    This is only useful when the dataset is not preprocessed to ignore
    the borders.

    Works for both segmentation and boundary maps.
    """

    def __init__(self, width=10, ignore_label=255):
        self.width = width
        self.label = ignore_label

    def __call__(self, results):
        for key in results.get("seg_fields", []):
            # TODO: should warn if label is in the segmentation map
            gt_seg = results[key]
            gt_seg[0 : self.width, :] = self.label
            gt_seg[-self.width :, :] = self.label
            gt_seg[:, 0 : self.width] = self.label
            gt_seg[:, -self.width :] = self.label
            results[key] = gt_seg
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(width={self.width})"


@PIPELINES.register_module(force=True)
class Resize(MMSEG_Resize):

    _supported_types = ("gt_semantic_seg", "gt_semantic_edge", "gt_inst_seg")

    def _resize_seg(self, results):
        for key in results.get("seg_fields", []):

            assert (
                key in self._supported_types
            ), f"ERR: {key} is not in {self._supported_types}"

            """Semantic edge:
            - should work fine for most cases
            - however, the edges might be thicker than how it was preprocessed
            """

            """Instance segmentation:
            - formatting matters for cv2 input
            - https://stackoverflow.com/questions/15245262/opencv-mat-element-types-and-their-sizes
            - 32-bit signed integer should work (CV_32S)
            """

            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results["scale"], interpolation="nearest"
                )
            else:
                gt_seg = mmcv.imresize(
                    results[key], results["scale"], interpolation="nearest"
                )
            results[key] = gt_seg


@PIPELINES.register_module(force=True)
class Pad(MMSEG_Pad):
    def __init__(
        self,
        size=None,
        size_divisor=None,
        pad_val=0,
        seg_pad_val=255,  # NOTE: 255 == background / ignore
        edge_pad_val=0,  # NOTE: 0 == background
        inst_pad_val=0,  # NOTE: 0 == background
    ):
        super().__init__(
            size=size,
            size_divisor=size_divisor,
            pad_val=pad_val,
            seg_pad_val=seg_pad_val,
        )
        self.edge_pad_val = edge_pad_val
        self.inst_pad_val = inst_pad_val

    def _pad_seg(self, results):
        for key in results.get("seg_fields", []):
            if key == "gt_semantic_edge":
                results[key] = mmcv.impad(
                    results[key],
                    shape=results["pad_shape"][:2],
                    pad_val=self.edge_pad_val,
                )
            elif key == "gt_inst_seg":
                results[key] = mmcv.impad(
                    results[key],
                    shape=results["pad_shape"][:2],
                    pad_val=self.inst_pad_val,
                )
            elif key == "gt_semantic_seg":
                results[key] = mmcv.impad(
                    results[key],
                    shape=results["pad_shape"][:2],
                    pad_val=self.seg_pad_val,
                )
            else:
                raise ValueError(f"{key} is not supported")


@PIPELINES.register_module(force=True)
class RandomRotate(MMSEG_RandomRotate):
    def __init__(
        self,
        prob,
        degree,
        pad_val=0,
        seg_pad_val=255,  # NOTE: 255 == background / ignore
        center=None,
        auto_bound=False,
        edge_pad_val=0,  # NOTE: 0 == background
        inst_pad_val=0,  # NOTE: 0 == background
    ):
        super().__init__(
            prob=prob,
            degree=degree,
            pad_val=pad_val,
            seg_pad_val=seg_pad_val,
            center=center,
            auto_bound=auto_bound,
        )
        self.edge_pad_val = (edge_pad_val,)
        self.inst_pad_val = inst_pad_val

    def __call__(self, results):
        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results["img"] = mmcv.imrotate(
                results["img"],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound,
            )

            # rotate segs
            for key in results.get("seg_fields", []):
                if key == "gt_semantic_edge":
                    results[key] = mmcv.imrotate(
                        results[key],
                        angle=degree,
                        border_value=self.edge_pad_val,
                        center=self.center,
                        auto_bound=self.auto_bound,
                        interpolation="nearest",
                    )
                elif key == "gt_inst_seg":
                    results[key] = mmcv.imrotate(
                        results[key],
                        angle=degree,
                        border_value=self.inst_pad_val,
                        center=self.center,
                        auto_bound=self.auto_bound,
                        interpolation="nearest",
                    )
                elif key == "gt_semantic_seg":
                    results[key] = mmcv.imrotate(
                        results[key],
                        angle=degree,
                        border_value=self.seg_pad_val,
                        center=self.center,
                        auto_bound=self.auto_bound,
                        interpolation="nearest",
                    )
                else:
                    raise ValueError(f"ERR: {key} is not supported")
        return results


@PIPELINES.register_module(force=True)
class RandomCrop(MMSEG_RandomCrop):
    """Random crop the image & seg & edge.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.get_crop_bbox(img)

        if self.cat_max_ratio < 1.0:
            if results.get("gt_semantic_seg", None) is not None:
                # Repeat 10 times
                for _ in range(10):
                    seg_temp = self.crop(results["gt_semantic_seg"], crop_bbox)
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_index]
                    if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                        break
                    crop_bbox = self.get_crop_bbox(img)
            else:
                # fall back to edge
                for _ in range(10):
                    edge_temp = self.crop(results["gt_semantic_edge"], crop_bbox)
                    labels, cnt = np.unique(edge_temp, return_counts=True)
                    if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                        break
                    crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"
