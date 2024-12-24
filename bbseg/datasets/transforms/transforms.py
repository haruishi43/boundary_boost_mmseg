#!/usr/bin/env python3

"""Custom transforms for joint segmentation and boundary detection.

Support for the following keys:
- gt_seg_map
- gt_inst_map
- gt_bin_edge
- gt_mlbl_edge

NOTE:
- gt_mlbl_edge, gt_bin_edge are also included in seg_fields (boundaries are dense pixels)
"""

import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmcv.transforms import (
    Resize as MMCV_Resize,
    # RandomResize as MMCV_RandomResize,
    Pad as MMCV_Pad,
)
from mmseg.datasets.transforms import (
    RandomCrop as MMSEG_RandomCrop,
    RandomRotate as MMSEG_RandomRotate,
)

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class Resize(MMCV_Resize):
    """Resize with edge keys.

    RandomResize should also use this method and I think TRANSFOMRS.build()
    should be able to handle this.

    TODO: I think mmseg's Resize is good enough
    """

    _seg_keys = ("gt_seg_map", "gt_inst_map")
    _edge_keys = ("gt_bin_edge", "gt_mlbl_edge")

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation maps with ``results['scale']``."""
        for key in self._seg_keys:
            if results.get(key, None) is not None:
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key],
                        results["scale"],
                        interpolation="nearest",
                        backend=self.backend,
                    )
                else:
                    gt_seg = mmcv.imresize(
                        results[key],
                        results["scale"],
                        interpolation="nearest",
                        backend=self.backend,
                    )
                results[key] = gt_seg

    def _resize_edge(self, results: dict) -> None:
        """Resize boundary maps with ``results['scale']``."""
        for key in self._edge_keys:
            if results.get(key, None) is not None:
                if self.keep_ratio:
                    gt_edge = mmcv.imrescale(
                        results[key],
                        results["scale"],
                        interpolation="nearest",
                        backend=self.backend,
                    )
                else:
                    gt_edge = mmcv.imresize(
                        results[key],
                        results["scale"],
                        interpolation="nearest",
                        backend=self.backend,
                    )
                results[key] = gt_edge

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._resize_edge(results)
        return results


@TRANSFORMS.register_module(force=True)
class Pad(MMCV_Pad):
    def __init__(
        self,
        pad_val=dict(img=0, seg=255, inst=0, edge=0),
        **kwargs,
    ):
        super().__init__(pad_val=pad_val, **kwargs)

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get("gt_seg_map", None) is not None:
            pad_val = self.pad_val.get("seg", 255)
            if isinstance(pad_val, int) and results["gt_seg_map"].ndim == 3:
                pad_val = tuple(pad_val for _ in range(results["gt_seg_map"].shape[2]))
            results["gt_seg_map"] = mmcv.impad(
                results["gt_seg_map"],
                shape=results["pad_shape"][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode,
            )
        if results.get("gt_inst_map", None) is not None:
            pad_val = self.pad_val.get("inst", 0)
            results["gt_inst_map"] = mmcv.impad(
                results["gt_inst_map"],
                shape=results["pad_shape"][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode,
            )
        for key in ("gt_bin_edge", "gt_mlbl_edge"):
            if results.get(key, None) is not None:
                pad_val = self.pad_val.get("edge", 0)
                results[key] = mmcv.impad(
                    results[key],
                    shape=results["pad_shape"][:2],
                    pad_val=pad_val,
                    padding_mode=self.padding_mode,
                )


@TRANSFORMS.register_module(force=True)
class RandomCrop(MMSEG_RandomCrop):
    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results["img"]
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            if results.get("gt_seg_map", None) is not None:
                # Repeat 10 times
                for _ in range(10):
                    seg_temp = self.crop(results["gt_seg_map"], crop_bbox)
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_index]
                    if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                        break
                    crop_bbox = generate_crop_bbox(img)
            else:
                if results.get("gt_mlbl_edge", None) is not None:
                    key = "gt_mlbl_edge"
                elif results.get("gt_bin_edge", None) is not None:
                    key = "gt_bin_edge"
                else:
                    raise ValueError(
                        "gt_seg_map, gt_mlbl_edge, gt_bin_edge are all None"
                    )
                # fall back to boundary
                for _ in range(10):
                    edge_temp = self.crop(results[key], crop_bbox)
                    labels, cnt = np.unique(edge_temp, return_counts=True)
                    if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                        break
                    crop_bbox = generate_crop_bbox(img)
        return crop_bbox


@TRANSFORMS.register_module(force=True)
class RandomRotate(MMSEG_RandomRotate):
    def __init__(
        self,
        prob,
        degree,
        pad_val=0,
        seg_pad_val=255,
        edge_pad_val=0,
        inst_pad_val=0,
        center=None,
        auto_bound=False,
    ) -> None:
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f"degree {degree} should be positive"
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, (
            f"degree {self.degree} should be a " f"tuple of (min, max)"
        )
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.edge_pad_val = edge_pad_val
        self.inst_pad_val = inst_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate, degree = self.generate_degree()
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
            if results.get("gt_seg_map", None) is not None:
                results["gt_seg_map"] = mmcv.imrotate(
                    results["gt_seg_map"],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation="nearest",
                )
            if results.get("gt_inst_map", None) is not None:
                results["gt_inst_map"] = mmcv.imrotate(
                    results["gt_inst_map"],
                    angle=degree,
                    border_value=self.inst_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation="nearest",
                )
            if results.get("gt_bin_edge", None) is not None:
                results["gt_bin_edge"] = mmcv.imrotate(
                    results["gt_bin_edge"],
                    angle=degree,
                    border_value=self.edge_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation="nearest",
                )
            if results.get("gt_mlbl_edge", None) is not None:
                results["gt_mlbl_edge"] = mmcv.imrotate(
                    results["gt_mlbl_edge"],
                    angle=degree,
                    border_value=self.edge_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation="nearest",
                )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(prob={self.prob}, "
            f"degree={self.degree}, "
            f"pad_val={self.pal_val}, "
            f"seg_pad_val={self.seg_pad_val}, "
            f"inst_pad_val={self.inst_pad_val}, "
            f"edge_pad_val={self.edge_pad_val}, "
            f"center={self.center}, "
            f"auto_bound={self.auto_bound})"
        )
        return repr_str


@TRANSFORMS.register_module()
class AddIgnoreBorder(BaseTransform):
    """Add ignore border to the semantic segmentation map.

    This is only useful when the dataset is not preprocessed to ignore
    the borders.

    Works for both segmentation and boundary maps.
    """

    def __init__(self, width: int = 10, ignore_label: int = 255) -> None:
        self.width = width
        self.label = ignore_label

    def transforms(self, results):
        for key in results.get("seg_fields", []):
            if key in ("gt_seg_map", "gt_inst_map"):
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
