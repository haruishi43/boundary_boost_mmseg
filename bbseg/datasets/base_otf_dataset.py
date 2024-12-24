#!/usr/bin/env python3

import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS


@DATASETS.register_module()
class BaseOTFEdgeDataset(BaseDataset):
    """Segmentation dataset for on-the-fly edge generation.

    Args:
        ann_file (str): Annotation file path.
        img_suffix (str): Image file suffix.
        seg_map_suffix (str): Segmentation map file suffix.
        inst_map_suffix (str): Instance map file suffix.
        inst_sensitive (bool): Whether to use instance sensitive edge.
        labelIds (Sequence): Label ids for semantic segmentation.
        inst_labelIds (Sequence): Label ids for instance segmentation.
        ignore_indices (Sequence): Ignore indices for edge generation.
        label2trainId (Dict): Mapping from label id to train id.
        metainfo (dict): Meta information of dataset.
        data_root (str): Data root for dataset.
        data_prefix (dict): Data prefix for dataset.
        filter_cfg (dict): Filter configuration for dataset.
        indices (Union[int, Sequence[int]]): Indices for dataset.
        serialize_data (bool): Whether to serialize data.
        pipeline (List[Union[dict, Callable]]): Pipeline for dataset.
        test_mode (bool): Whether in test mode.
        lazy_init (bool): Whether to lazy initialize the dataset.
        max_refetch (int): Maximum number of refetching data.
        ignore_index (int): Ignore index for edge generation.
        reduce_zero_label (bool): Whether to reduce zero label.
        backend_args (dict): Backend arguments for dataset.
    """

    METAINFO: dict = dict()

    def __init__(
        self,
        ann_file: str = "",
        img_suffix: str = ".jpg",
        seg_map_suffix: str = ".png",
        inst_map_suffix: str = "_inst.png",
        inst_sensitive: bool = False,
        labelIds: Optional[Sequence] = None,
        inst_labelIds: Optional[Sequence] = None,
        ignore_indices: Optional[Sequence] = [],
        label2trainId: Optional[Dict] = None,
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(
            img_path="",
            seg_map_path="",
        ),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        ignore_index: int = 255,
        reduce_zero_label: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        if inst_sensitive:
            # assume that instance map is located in the same directory as seg map
            assert seg_map_suffix != inst_map_suffix
            assert inst_map_suffix is not None
        self.inst_map_suffix = inst_map_suffix
        self.inst_sensitive = inst_sensitive

        assert labelIds is not None, "labelIds should be specified"
        self.labelIds = labelIds
        self.inst_labelIds = inst_labelIds
        self.ignore_indices = ignore_indices
        self.label2trainId = label2trainId

        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get("classes", None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(label_map=self.label_map, reduce_zero_label=self.reduce_zero_label)
        )

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert (
                self._metainfo.get("classes") is not None
            ), "dataset metainfo `classes` should be specified when testing"

    @classmethod
    def get_label_map(cls, new_classes: Optional[Sequence] = None) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get("classes", None)
        if (
            new_classes is not None
            and old_classes is not None
            and list(new_classes) != list(old_classes)
        ):
            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO["classes"]):
                raise ValueError(
                    f"new classes {new_classes} is not a "
                    f"subset of classes {old_classes} in METAINFO."
                )
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get("palette", [])
        classes = self._metainfo.get("classes", [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError(
                "palette does not match classes " f"as metainfo is {self._metainfo}."
            )
        return new_palette

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(
                self.ann_file
            ), f"Failed to load `ann_file` {self.ann_file}"
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args
            )
            for line in lines:
                img_name = line.strip()
                data_info = dict(img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                    if self.inst_sensitive:
                        inst_map = img_name + self.inst_map_suffix
                        data_info["inst_map_path"] = osp.join(ann_dir, inst_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["inst_sensitive"] = self.inst_sensitive
                data_info["labelIds"] = self.labelIds
                data_info["inst_labelIds"] = self.inst_labelIds
                data_info["ignore_indices"] = self.ignore_indices
                data_info["label2trainId"] = self.label2trainId
                data_info["seg_fields"] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args,
            ):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                    if self.inst_sensitive:
                        inst_map = img.replace(self.img_suffix, self.inst_map_suffix)
                        data_info["inst_map_path"] = osp.join(ann_dir, inst_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["inst_sensitive"] = self.inst_sensitive
                data_info["labelIds"] = self.labelIds
                data_info["inst_labelIds"] = self.inst_labelIds
                data_info["ignore_indices"] = self.ignore_indices
                data_info["label2trainId"] = self.label2trainId
                data_info["seg_fields"] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x["img_path"])
        return data_list
