#!/usr/bin/env python3

from mmengine.structures import PixelData
from mmseg.structures.seg_data_sample import SegDataSample


class BBSegDataSample(SegDataSample):

    @property
    def gt_bb_seg(self) -> PixelData:
        return self._gt_bb_seg

    @gt_bb_seg.setter
    def gt_bb_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_bb_seg', dtype=PixelData)

    @gt_bb_seg.deleter
    def gt_bb_seg(self) -> None:
        del self._gt_bb_seg
