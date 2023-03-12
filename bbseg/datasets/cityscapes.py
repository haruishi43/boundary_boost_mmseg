#!/usr/bin/env python3

from .builder import DATASETS
from .custom import OTFCustomJointDataset

from pyEdgeEval.datasets.cityscapes_attributes import (
    CITYSCAPES_inst_labelIds,
    CITYSCAPES_labelIds,
    CITYSCAPES_label2trainId,
)

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
class OTFJointCityscapesDataset(OTFCustomJointDataset):
    """OTF edge generation for Cityscapes dataset

    - should only be used for training the model
    """

    CLASSES = CITYSCAPES_CLASSES
    PALETTE = CITYSCAPES_PALETTE

    def __init__(
        self,
        img_suffix="_leftImg8bit.png",
        seg_map_suffix="_gtFine_labelIds.png",
        inst_map_suffix="_gtFine_instanceIds.png",
        inst_sensitive=True,  # default instance sensitive
        labelIds=CITYSCAPES_labelIds,
        inst_labelIds=CITYSCAPES_inst_labelIds,
        ignore_indicies=[2, 3],
        label2trainId=CITYSCAPES_label2trainId,
        **kwargs,
    ):
        super(OTFJointCityscapesDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            inst_map_suffix=inst_map_suffix,
            inst_sensitive=inst_sensitive,
            labelIds=labelIds,
            inst_labelIds=inst_labelIds,
            ignore_indices=ignore_indicies,
            label2trainId=label2trainId,
            **kwargs,
        )

    def format_results(self, **kwargs):
        raise ValueError("ERR: Should not use OTF for test set")

    def evaluate(self, **kwargs):
        raise ValueError("ERR: Should not use OTF for evaluation!")
