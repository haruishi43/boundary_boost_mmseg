#!/usr/bin/env python3

from mmseg.models.builder import (
    MODELS,
    BACKBONES,
    NECKS,
    HEADS,
    SEGMENTORS,
    LOSSES,
    build_backbone,
    build_neck,
    build_head,
    build_loss,
    build_segmentor,
)

__all__ = [
    "MODELS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "SEGMENTORS",
    "LOSSES",
    "build_backbone",
    "build_neck",
    "build_head",
    "build_loss",
    "build_segmentor",
]
