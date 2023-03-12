#!/usr/bin/env python3

"""Custom Joint Dataset Classes."""

import os.path as osp

import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from pyEdgeEval.edge_tools import Mask2Edge, InstanceMask2Edge

from .base_dataset import BaseDataset
from .pipelines import (
    Compose,
    LoadAnnotations,
    EdgeFormatBundle,
    BinaryEdgeFormatBundle,
    JointFormatBundle,
    JointBinaryFormatBundle,
)


class OTFCustomJointDataset(BaseDataset):
    """Custom dataset class where we generate edges OTF"""

    inst_sensitive = None
    mask2edge = None

    def __init__(
        self,
        inst_dir=None,
        inst_map_suffix="_inst.png",
        inst_sensitive=False,
        gt_seg_loader_cfg=None,
        gt_edge_loader_cfg=None,
        labelIds=None,
        inst_labelIds=None,
        ignore_indices=[],
        label2trainId=None,
        radius=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inst_dir = inst_dir
        self.inst_map_suffix = inst_map_suffix
        self.inst_sensitive = inst_sensitive

        # initialize mask2edge
        assert labelIds is not None, "labelIds is None"
        labelIds = labelIds
        if self.inst_sensitive:
            assert (
                inst_labelIds
            ), "ERR: `inst_labelIds` is needed for instance sensitive OTF"
            self.mask2edge = InstanceMask2Edge(
                labelIds=labelIds,
                inst_labelIds=inst_labelIds,
                ignore_indices=ignore_indices,
                label2trainId=label2trainId,
                radius=radius,
                use_cv2=True,
                quality=0,
            )
        else:
            self.mask2edge = Mask2Edge(
                labelIds=labelIds,
                ignore_indices=ignore_indices,
                label2trainId=label2trainId,
                radius=radius,
                use_cv2=True,
                quality=0,
            )

        self.gt_seg_loader = Compose(
            [
                LoadAnnotations()
                if gt_seg_loader_cfg is None
                else LoadAnnotations(**gt_seg_loader_cfg),
                JointFormatBundle(),  # need to convert labels to trainIds
            ]
        )

        self.gt_edge_loader = Compose(
            [
                LoadAnnotations()
                if gt_edge_loader_cfg is None
                else LoadAnnotations(**gt_edge_loader_cfg),
                EdgeFormatBundle(),  # need to convert seg to edges
            ]
        )

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.inst_dir is None or osp.isabs(self.inst_dir)):
                self.inst_dir = osp.join(self.data_root, self.inst_dir)

        # load annotations
        assert self.ann_dir is not None
        self.img_infos = self.load_annotations(
            img_dir=self.img_dir,
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
            inst_map_suffix=self.inst_map_suffix,
            inst_sensitive=self.inst_sensitive,
            split=self.split,
        )

    def load_annotations(
        self,
        img_dir,
        img_suffix,
        seg_map_suffix,
        inst_map_suffix,
        inst_sensitive,
        split,
    ):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    seg_map = img_name + seg_map_suffix
                    if inst_sensitive:
                        # need instance map for instance aware segmentation (otf)
                        inst_map = img_name + inst_map_suffix
                        ann = dict(
                            seg_map=seg_map,
                            inst_map=inst_map,
                        )
                    else:
                        ann = dict(
                            seg_map=seg_map,
                        )
                    img_info["ann"] = ann
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                seg_map = img.replace(img_suffix, seg_map_suffix)
                if inst_sensitive:
                    inst_map = img.replace(img_suffix, inst_map_suffix)
                    ann = dict(
                        seg_map=seg_map,
                        inst_map=inst_map,
                    )
                else:
                    ann = dict(
                        seg_map=seg_map,
                    )
                img_info["ann"] = ann
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir
        results["seg_prefix"] = self.ann_dir
        results["inst_prefix"] = self.inst_dir
        results["inst_sensitive"] = self.inst_sensitive
        results["mask2edge"] = self.mask2edge


class OTFCustomBinaryJointDataset(BaseDataset):

    mask2edge = None

    def __init__(
        self,
        gt_seg_loader_cfg=None,
        gt_edge_loader_cfg=None,
        ignore_indices=[],
        labelIds=None,
        label2trainId=None,
        radius=1,
        selected_label=1,  # index for the edge
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert labelIds is not None
        labelIds = labelIds
        self.mask2edge = Mask2Edge(
            labelIds=labelIds,
            ignore_indices=ignore_indices,
            label2trainId=label2trainId,
            radius=radius,
            use_cv2=True,
            quality=0,
        )

        self.gt_seg_loader = Compose(
            [
                LoadAnnotations()
                if gt_seg_loader_cfg is None
                else LoadAnnotations(**gt_seg_loader_cfg),
                JointBinaryFormatBundle(selected_label=selected_label),
            ]
        )

        assert selected_label is not None
        self.gt_edge_loader = Compose(
            [
                LoadAnnotations()
                if gt_edge_loader_cfg is None
                else LoadAnnotations(**gt_edge_loader_cfg),
                BinaryEdgeFormatBundle(
                    selected_label=selected_label,
                ),
            ]
        )

        # load annotations
        assert self.ann_dir is not None
        self.img_infos = self.load_annotations(
            img_dir=self.img_dir,
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
            split=self.split,
        )

    def load_annotations(
        self,
        img_dir,
        img_suffix,
        seg_map_suffix,
        split,
    ):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    seg_map = img_name + seg_map_suffix
                    img_info["ann"] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir
        results["seg_prefix"] = self.ann_dir
        results["mask2edge"] = self.mask2edge
