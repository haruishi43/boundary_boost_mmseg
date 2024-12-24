#!/usr/bin/env python3

from typing import List, Optional

import numpy as np
from mmcv.transforms.base import BaseTransform
from pyEdgeEval.utils import (
    mask2onehot,
    edge_label2trainId,
)
from pyEdgeEval.edge_tools.transforms import mask2edge as _mask2edge

from mmseg.registry import TRANSFORMS


def mask2edge_nonIS(
    mask: np.ndarray,
    labelIds: List[int],
    label2trainId: Optional[dict] = None,
    ignore_indices: List[int] = [],
    radius: int = 2,
    use_cv2: bool = True,
    quality: int = 0,
) -> np.ndarray:
    """Convert Mask to Edge."""
    assert mask.ndim == 2
    onehot_mask = mask2onehot(mask, labels=labelIds)
    edge = _mask2edge(
        run_type="loop",
        instance_sensitive=False,
        mask=onehot_mask,
        ignore_indices=ignore_indices,
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )
    if label2trainId:
        edge = edge_label2trainId(edge=edge, label2trainId=label2trainId)
    return edge


def mask2edge(
    mask: np.ndarray,
    inst_mask: np.ndarray,
    labelIds: List[int],
    inst_labelIds: List[int],
    label2trainId: Optional[dict] = None,
    ignore_indices: List[int] = [],
    radius: int = 2,
    use_cv2: bool = True,
    quality: int = 0,
) -> np.ndarray:
    """Convert Mask to Instance Sensitive Edge."""
    assert mask.ndim == 2
    assert mask.shape == inst_mask.shape
    onehot_mask = mask2onehot(mask, labels=labelIds)
    edge = _mask2edge(
        run_type="loop",
        instance_sensitive=True,
        mask=onehot_mask,
        inst_mask=inst_mask,
        inst_labelIds=inst_labelIds,
        ignore_indices=ignore_indices,
        radius=radius,
        use_cv2=use_cv2,
        quality=quality,
    )
    if label2trainId:
        edge = edge_label2trainId(edge=edge, label2trainId=label2trainId)
    return edge


def mask_label2trainId(
    mask: np.ndarray,
    label2trainId: dict,
    ignore_index: int = 255,
) -> np.ndarray:
    """Python version of `labelid2trainid` function for segmentation data

    Args:
        mask: single channel image containing segmentation label

    Returns:
        np.ndarray
    """

    if len(mask.shape) == 2:
        h, w = mask.shape
    elif len(mask.shape) == 3:
        h, w, c = mask.shape
        assert c == 1, f"ERR: input label has {c} channels which should be 1"
    else:
        raise ValueError()

    # make a new array with ignore_index
    new_mask = np.full((h, w), ignore_index, dtype=np.uint32)

    for lbl, t_id in label2trainId.items():
        m = mask == lbl
        new_mask[m] = t_id

    return new_mask


@TRANSFORMS.register_module()
class DecodeMultiLabelEdge(BaseTransform):
    """Decode RGB-file to 24bit array edges.

    This is required when loading edges from file instead of OTF generation.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def transform(self, results: dict) -> dict:
        edge = results.get("gt_mlbl_edge", None)
        assert edge is not None, "ERR: gt_sem_edge is not available"
        assert edge.ndim == 3, "need to be 3-dim image"
        assert edge.shape[-1] == 3, "not rgb gt"

        if results.get("num_classes", None) is not None:
            # results["num_classses"] is used for backward compatibility
            _num_classes = results["num_classes"]
            assert self.num_classes == _num_classes, (
                f"ERR: num_classes mismatch, "
                f"got {self.num_classes} and {_num_classes}"
            )

        # HACK: decode RGB to 24bit array
        # it's only possible to encode 24 classes
        edge = np.unpackbits(
            edge,
            axis=2,
        )[:, :, -1 : -(self.num_classes + 1) : -1]

        # transpose to (C, H, W)
        edge = np.ascontiguousarray(edge.transpose(2, 0, 1))

        # set multi-label edge
        results["gt_mlbl_edge"] = edge

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(num_classes={self.num_classes})"
        return repr_str


@TRANSFORMS.register_module()
class OTFSeg2MultiLabelEdge(BaseTransform):
    """Generates multi-label edge from segmentation map."""

    def __init__(
        self,
        radius: int = 2,
        use_cv2: bool = True,
        quality: int = 0,
    ) -> None:
        self._mask2edge_kwargs = dict(
            radius=radius,
            use_cv2=use_cv2,
            quality=quality,
        )

    def transform(self, results: dict) -> dict:
        mask = results.get("gt_seg_map", None)
        assert mask is not None, "ERR: gt_seg_map is not available"

        labelIds = results.get("labelIds", None)
        assert labelIds is not None, "ERR: labelIds is not available"
        label2trainId = results.get("label2trainId", None)
        ignore_indices = results.get("ignore_indices", [])

        if results.get("inst_sensitive", False):
            inst_map = results.get("gt_inst_map", None)
            assert inst_map is not None, "ERR: instance map is not available"
            inst_labelIds = results.get("inst_labelIds", None)
            assert inst_labelIds is not None, "ERR: inst_labelIds is not available"

            edges = mask2edge(
                mask=mask,
                inst_mask=inst_map,
                labelIds=labelIds,
                inst_labelIds=inst_labelIds,
                label2trainId=label2trainId,
                ignore_indices=ignore_indices,
                **self._mask2edge_kwargs,
            )
        else:
            edges = mask2edge_nonIS(
                mask=mask,
                labelIds=labelIds,
                label2trainId=label2trainId,
                ignore_indices=ignore_indices,
                **self._mask2edge_kwargs,
            )

        if results.get("reduce_zero_label", False):
            # reduce zero label
            edges = edges[1:]

        # set multi-label edge
        results["gt_mlbl_edge"] = edges

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mask2edge_kwargs={self._mask2edge_kwargs})"
        return repr_str


@TRANSFORMS.register_module()
class OTFSeg2BinaryEdge(BaseTransform):
    """Generates binary edge from segmentation map.

    This is useful when the model requires auxiliary binary edge supervision
    with different radius.
    """

    def __init__(
        self,
        radius: int = 2,
        use_cv2: bool = True,
        quality: int = 0,
    ) -> None:
        self._mask2edge_kwargs = dict(
            radius=radius,
            use_cv2=use_cv2,
            quality=quality,
        )

    def transform(self, results: dict) -> dict:
        mask = results.get("gt_seg_map", None)
        assert mask is not None, "ERR: gt_seg_map is not available"

        labelIds = results.get("labelIds", None)
        assert labelIds is not None, "ERR: labelIds is not available"
        label2trainId = results.get("label2trainId", None)
        ignore_indices = results.get("ignore_indices", [])

        if results.get("inst_sensitive", False):
            inst_map = results.get("gt_inst_map", None)
            assert inst_map is not None, "ERR: instance map is not available"
            inst_labelIds = results.get("inst_labelIds", None)
            assert inst_labelIds is not None, "ERR: inst_labelIds is not available"

            edges = mask2edge(
                mask=mask,
                inst_mask=inst_map,
                labelIds=labelIds,
                inst_labelIds=inst_labelIds,
                label2trainId=label2trainId,
                ignore_indices=ignore_indices,
                **self._mask2edge_kwargs,
            )
        else:
            edges = mask2edge_nonIS(
                mask=mask,
                labelIds=labelIds,
                label2trainId=label2trainId,
                ignore_indices=ignore_indices,
                **self._mask2edge_kwargs,
            )

        if results.get("reduce_zero_label", False):
            # reduce zero label
            edges = edges[1:]

        # to binary
        edges = (edges.sum(axis=0) > 0).astype(np.uint8)

        # set multi-edge
        results["gt_bin_edge"] = edges

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mask2edge_kwargs={self._mask2edge_kwargs})"
        return repr_str


@TRANSFORMS.register_module()
class FormatSegMask(BaseTransform):
    """Format segmentation mask using label2trainId.

    This is useful when the model requires segmentation mask with
    different label mapping.
    """

    def transform(self, results: dict) -> dict:
        mask = results.get("gt_seg_map", None)
        assert mask is not None, "ERR: gt_seg_map is not available"

        label2trainId = results.get("label2trainId", None)
        if label2trainId:
            mask = mask_label2trainId(mask, label2trainId)

            if results.get("reduce_zero_label", False):
                # avoid using underflow conversion
                mask[mask == 0] = 255
                mask = mask - 1
                mask[mask == 254] = 255

            # modify if custom classes
            if results.get("label_map", None) is not None:
                # Add deep copy to solve bug of repeatedly
                # replace `gt_semantic_seg`, which is reported in
                # https://github.com/open-mmlab/mmsegmentation/pull/1445/
                mask_copy = mask.copy()
                for old_id, new_id in results["label_map"].items():
                    mask[mask_copy == old_id] = new_id

            results["gt_seg_map"] = mask

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class GenerateBinaryFromMultiLabel(BaseTransform):
    """Generate binary edge from multi-label edge."""

    def transform(self, results: dict) -> dict:
        mlbl_edge = results.get("gt_mlbl_edge", None)
        assert mlbl_edge is not None, "ERR: gt_mlbl_edge is not available"

        bin_edge = (mlbl_edge.sum(axis=0) > 0).astype(np.uint8)
        results["gt_bin_edge"] = bin_edge

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
