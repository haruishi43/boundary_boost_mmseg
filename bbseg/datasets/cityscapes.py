#!/usr/bin/env python3

from typing import Dict, Optional, Sequence

from mmseg.registry import DATASETS
from pyEdgeEval.datasets.cityscapes_attributes import (
    CITYSCAPES_inst_labelIds,
    CITYSCAPES_labelIds,
    CITYSCAPES_label2trainId,
)

from .base_joint_dataset import BaseSegEdgeDataset
from .base_otf_dataset import BaseOTFEdgeDataset


CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
CITYSCAPES_PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]
ignore_ids = [2, 3]


@DATASETS.register_module()
class CityscapesSegEdgeDataset(BaseSegEdgeDataset):
    """Cityscapes dataset with semseg and edge detection.

    Currently, I don't preprocess the boundaries as binary edge masks, so
    I load the multi-label edge masks. If you want binary edge masks, you can
    use the `GenerateBinaryFromMultiLabel` transform.
    """

    METAINFO = dict(
        classes=CITYSCAPES_CLASSES,
        palette=CITYSCAPES_PALETTE,
    )

    def __init__(
        self,
        img_suffix: str = "_leftImg8bit.png",
        seg_map_suffix="_gtFine_labelTrainIds.png",
        inst_sensitive: bool = True,
        thin: bool = False,
        edge_mode: str = "mlbl",
        test_mode: bool = False,
        **kwargs,
    ) -> None:
        if inst_sensitive:
            if test_mode:
                if thin:
                    mlbl_edge_suffix = "_gtProc_thin_isedge.png"
                else:
                    mlbl_edge_suffix = "_gtProc_raw_isedge.png"
            else:
                mlbl_edge_suffix = "_gtProc_isedge.png"
        else:
            if test_mode:
                if thin:
                    mlbl_edge_suffix = "_gtProc_thin_edge.png"
                else:
                    mlbl_edge_suffix = "_gtProc_raw_edge.png"
            else:
                mlbl_edge_suffix = "_gtProc_edge.png"

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            mlbl_edge_suffix=mlbl_edge_suffix,
            edge_mode=edge_mode,
            test_mode=test_mode,
            **kwargs,
        )


@DATASETS.register_module()
class CityscapesOTFEdgeDataset(BaseOTFEdgeDataset):
    """OTF edge generation for Cityscapes dataset."""

    METAINFO = dict(
        classes=CITYSCAPES_CLASSES,
        palette=CITYSCAPES_PALETTE,
    )

    def __init__(
        self,
        img_suffix: str = "_leftImg8bit.png",
        seg_map_suffix: str = "_gtFine_labelIds.png",
        inst_map_suffix: str = "_gtFine_instanceIds.png",
        inst_sensitive: bool = True,
        labelIds: Optional[Sequence] = CITYSCAPES_labelIds,
        inst_labelIds: Optional[Sequence] = CITYSCAPES_inst_labelIds,
        ignore_indices: Optional[Sequence] = ignore_ids,
        label2trainId: Optional[Dict] = CITYSCAPES_label2trainId,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            inst_map_suffix=inst_map_suffix,
            inst_sensitive=inst_sensitive,
            labelIds=labelIds,
            inst_labelIds=inst_labelIds,
            ignore_indices=ignore_indices,
            label2trainId=label2trainId,
            **kwargs,
        )
