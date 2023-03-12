#!/usr/bin/env python3

"""builder.py

Currently a placeholder for subclassing `mmseg`.
"""

# I want to use the original segmentation only datasets
from mmseg.datasets import (
    DATASETS,
    PIPELINES,
    build_dataloader,
    build_dataset,
)

__all__ = [
    "DATASETS",
    "PIPELINES",
    "build_dataloader",
    "build_dataset",
]
