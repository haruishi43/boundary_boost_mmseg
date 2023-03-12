#!/usr/bin/env python3

"""Loading data (images and annotations)."""

import os.path as osp

import numpy as np

import mmcv
from mmseg.datasets.pipelines.loading import LoadAnnotations as MMSEG_LoadAnnotations

from ..builder import PIPELINES


@PIPELINES.register_module(force=True)
class LoadAnnotations(MMSEG_LoadAnnotations):
    """Extended `LoadAnnotations` class from mmseg.

    Added support for loading instance masks for instance-aware semantic boundaries.
    """

    def __call__(self, results):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("seg_prefix", None) is not None:
            filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
        else:
            filename = results["ann_info"]["seg_map"]
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = (
            mmcv.imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend)
            .squeeze()
            .astype(np.uint8)
        )
        # modify if custom classes (NOTE: currently not being used)
        if results.get("label_map", None) is not None:
            for old_id, new_id in results["label_map"].items():
                # Add deep copy to solve bug of repeatedly replacing
                # gt_semantic_seg
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        # reduce zero_label
        # It is better to reduce zero label after edges are computed.
        # Unless the it is necessary that the zero label is removed
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results["gt_semantic_seg"] = gt_semantic_seg
        results["seg_fields"].append("gt_semantic_seg")

        # load instance map
        inst_sensitive = results.get("inst_sensitive", False)
        if inst_sensitive:
            # load instance segmentation image
            inst_prefix = results.get("inst_prefix", None)
            if inst_prefix is not None:
                filename = osp.join(inst_prefix, results["ann_info"]["inst_map"])
            else:
                filename = results["ann_info"]["inst_map"]
            img_bytes = self.file_client.get(filename)
            gt_inst_seg = (
                mmcv.imfrombytes(
                    img_bytes, flag="unchanged", backend=self.imdecode_backend
                )
                .squeeze()
                .astype(np.int32)  # NOTE: needs to be int32
            )
            results["gt_inst_seg"] = gt_inst_seg
            results["seg_fields"].append("gt_inst_seg")

        return results
