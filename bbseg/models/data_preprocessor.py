#!/usr/bin/env python3


from mmseg.registry import MODELS
from mmseg.models import SegDataPreProcessor


@MODELS.register_module()
class BBSegDataPreProcessor(SegDataPreProcessor):

    def __init__(
        self,
        **kwargs,
    ):
        ...
