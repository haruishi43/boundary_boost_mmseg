#!/usr/bin/env python3

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadOTFSegAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    Args:
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        backend_args=None,
        imdecode_backend="pillow",
        inst_sensitive=False,
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
        )
        self.imdecode_backend = imdecode_backend
        self.inst_sensitive = inst_sensitive

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)
        gt_semantic_seg = (
            mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend)
            .squeeze()
            .astype(np.uint8)
        )
        # NOTE: do reduce_zero_label and label_map after generating edges
        # if results.get("reduce_zero_label", False):
        #     # avoid using underflow conversion
        #     gt_semantic_seg[gt_semantic_seg == 0] = 255
        #     gt_semantic_seg = gt_semantic_seg - 1
        #     gt_semantic_seg[gt_semantic_seg == 254] = 255
        # # modify if custom classes
        # if results.get("label_map", None) is not None:
        #     # Add deep copy to solve bug of repeatedly
        #     # replace `gt_semantic_seg`, which is reported in
        #     # https://github.com/open-mmlab/mmsegmentation/pull/1445/
        #     gt_semantic_seg_copy = gt_semantic_seg.copy()
        #     for old_id, new_id in results["label_map"].items():
        #         gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results["gt_seg_map"] = gt_semantic_seg
        results["seg_fields"].append("gt_seg_map")

        # load instance map
        if self.inst_sensitive or results.get("inst_sensitive", False):
            img_bytes = fileio.get(
                results["inst_map_path"], backend_args=self.backend_args
            )
            gt_inst_map = (
                mmcv.imfrombytes(
                    img_bytes,
                    flag="unchanged",
                    backend=self.imdecode_backend,
                )
                .squeeze()
                .astype(np.int32)  # NOTE: needs to be int32
            )
            results["gt_inst_map"] = gt_inst_map
            results["seg_fields"].append("gt_inst_map")

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str


@TRANSFORMS.register_module()
class LoadMultiLabelEdge(BaseTransform):
    def __init__(
        self,
        backend_args=None,
        imdecode_backend="pillow",
    ) -> None:
        self.imdecode_backend = imdecode_backend

        self.backend_args = None
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_edge(self, results: dict) -> None:
        img_bytes = fileio.get(
            results["mlbl_edge_path"],
            backend_args=self.backend_args,
        )

        # NOTE: only supports RGB inputs (need to change to binary or tif for more classes)
        # NOTE: for tif, we need to implement 32bit transforms
        gt_mlbl_edge = (
            mmcv.imfrombytes(
                img_bytes,
                flag="color",
                channel_order="rgb",
                backend=self.imdecode_backend,
            )
            .squeeze()
            .astype(np.uint8)
        )
        results["gt_mlbl_edge"] = gt_mlbl_edge
        results["seg_fields"].append("gt_mlbl_edge")

    def transform(self, results: dict) -> dict:
        self._load_edge(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str


@TRANSFORMS.register_module()
class LoadBinaryEdge(BaseTransform):
    def __init__(
        self,
        backend_args=None,
        imdecode_backend="pillow",
    ) -> None:
        self.imdecode_backend = imdecode_backend

        self.backend_args = None
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_edge(self, results: dict) -> None:
        img_bytes = fileio.get(
            results["bin_edge_path"],
            backend_args=self.backend_args,
        )

        # FIXME: uint8 might be great for transforming, but might not be
        # the original format
        gt_bin_edge = (
            mmcv.imfrombytes(
                img_bytes,
                flag="unchanged",
                backend=self.imdecode_backend,
            )
            .squeeze()
            .astype(np.uint8)
        )
        results["gt_bin_edge"] = gt_bin_edge
        results["seg_fields"].append("gt_bin_edge")

    def transform(self, results: dict) -> dict:
        self._load_edge(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str
