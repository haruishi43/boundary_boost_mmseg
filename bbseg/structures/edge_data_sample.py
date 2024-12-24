#!/usr/bin/env python3

"""Custom Data Sample.

SegDataSample + Boundary data.
"""

from mmengine.structures import PixelData
from mmseg.structures.seg_data_sample import SegDataSample


class SegEdgeDataSample(SegDataSample):

    # Multi-Label (Semantic) Edge

    @property
    def gt_mlbl_edge(self) -> PixelData:
        return self._gt_mlbl_edge

    @gt_mlbl_edge.setter
    def gt_mlbl_edge(self, value: PixelData) -> None:
        self.set_field(value, "_gt_mlbl_edge", dtype=PixelData)

    @gt_mlbl_edge.deleter
    def gt_mlbl_edge(self) -> None:
        del self._gt_mlbl_edge

    # Binary Edge

    @property
    def gt_bin_edge(self) -> PixelData:
        return self._gt_bin_edge

    @gt_bin_edge.setter
    def gt_bin_edge(self, value: PixelData) -> None:
        self.set_field(value, "_gt_bin_edge", dtype=PixelData)

    @gt_bin_edge.deleter
    def gt_bin_edge(self) -> None:
        del self._gt_bin_edge
