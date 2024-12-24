#!/usr/bin/env python3

import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from bbseg.structures import SegEdgeDataSample


@TRANSFORMS.register_module()
class PackSegEdgeInputs(BaseTransform):
    """Pack the inputs data for the binary edge inputs.

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``BasciEdgeDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
        selected_label (Optional[int]): if multi-label edges are given,
            the label set with this argument will be used (assuming
            that the edge is indexed from 0)
    """

    def __init__(
        self,
        meta_keys=(
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "reduce_zero_label",
        ),
    ) -> None:
        self.meta_keys = meta_keys

        # TODO: add more meta_keys

    def init_data_sample(self) -> SegEdgeDataSample:
        return SegEdgeDataSample()

    def pack_inputs(self, results: dict) -> dict:
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img
        return packed_results

    def get_seg_map(
        self, results: dict, data_sample: SegEdgeDataSample
    ) -> SegEdgeDataSample:
        if results.get("gt_seg_map", None) is not None:
            if len(results["gt_seg_map"].shape) == 2:
                data = to_tensor(results["gt_seg_map"][None, ...].astype(np.int64))
            else:
                warnings.warn(
                    "Please pay attention your ground truth "
                    "segmentation map, usually the segmentation "
                    "map is 2D, but got "
                    f'{results["gt_seg_map"].shape}'
                )
                data = to_tensor(results["gt_seg_map"].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)
        return data_sample

    def get_edge(
        self, results: dict, data_sample: SegEdgeDataSample
    ) -> SegEdgeDataSample:
        # get multi-label edge
        if results.get("gt_mlbl_edge", None) is not None:
            mlbl_edge = results["gt_mlbl_edge"]
            assert mlbl_edge is not None, "ERR: gt_mlbl_edge is not available"
            assert mlbl_edge.ndim == 3, "need to be 3-dim image"
            data = to_tensor(mlbl_edge.astype(np.int64))
            gt_mlbl_edge_data = dict(data=data)
            data_sample.gt_mlbl_edge = PixelData(**gt_mlbl_edge_data)

        # get binary-label edge
        if results.get("gt_bin_edge", None) is not None:
            bin_edge = results["gt_bin_edge"]
            assert bin_edge is not None, "ERR: gt_bin_edge is not available"
            assert bin_edge.ndim == 2
            data = to_tensor(bin_edge[None, ...].astype(np.int64))
            gt_bin_edge_data = dict(data=data)
            data_sample.gt_bin_edge = PixelData(**gt_bin_edge_data)

        return data_sample

    def set_img_metadata(
        self, results: dict, data_sample: SegEdgeDataSample
    ) -> SegEdgeDataSample:
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        return data_sample

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """

        # pack the inputs
        packed_results = self.pack_inputs(results)

        data_sample = self.init_data_sample()

        # get segmentation map
        data_sample = self.get_seg_map(results, data_sample)

        # get edge
        data_sample = self.get_edge(results, data_sample)

        # get metadata
        data_sample = self.set_img_metadata(results, data_sample)

        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
