#!/usr/bin/env python3

import numpy as np

from mmcv.parallel import DataContainer as DC
from mmseg.datasets.pipelines.formatting import (
    ToDataContainer as MMSEG_ToDataContainer,
    to_tensor,
)

from ..builder import PIPELINES


@PIPELINES.register_module(force=True)
class ToDataContainer(MMSEG_ToDataContainer):
    def __init__(
        self,
        fields=(
            dict(key="img", stack=True),
            dict(key="gt_semantic_seg"),
            dict(key="gt_semantic_edge"),
        ),
    ):
        self.fields = fields


@PIPELINES.register_module()
class FormatImage(object):
    def __call__(self, results):
        # input image
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results["img"] = DC(to_tensor(img), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class FormatEdge(object):
    """Format multi-label edge."""

    def __init__(self, reduce_zero_label=False):
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):
        # non-otf
        if "gt_semantic_edge" in results:
            edge = results["gt_semantic_edge"]

            assert edge.ndim == 3, "need to be 3-dim image"
            assert edge.shape[-1] == 3, "not rgb gt"

            # HACK: decode RGB to 24bit array
            # it's only possible to encode 24 classes
            edge = np.unpackbits(
                edge,
                axis=2,
            )[:, :, -1 : -(results["num_classes"] + 1) : -1]
            edge = np.ascontiguousarray(edge.transpose(2, 0, 1))

            # convert to long
            results["gt_semantic_edge"] = DC(
                to_tensor(edge.astype(np.int64)),
                stack=True,
            )
        elif "gt_semantic_seg" in results:
            mask2edge = results.get("mask2edge", None)
            assert mask2edge, "ERR: no mask2edge inside `results`"

            if results["inst_sensitive"]:
                inst_map = results.get("gt_inst_seg", None)
                assert inst_map is not None, "ERR: instance map is not available"
                out = mask2edge(
                    mask=results["gt_semantic_seg"],
                    inst_mask=inst_map,
                )
                # remove it from memory?
                del results["gt_inst_seg"]
            else:
                out = mask2edge(mask=results["gt_semantic_seg"])

            edge = out["edge"]
            assert edge.ndim == 3

            if self.reduce_zero_label:
                edge = edge[1:]

            results["gt_semantic_edge"] = DC(
                to_tensor(edge.astype(np.int64)),
                stack=True,
            )

        return results


@PIPELINES.register_module()
class FormatBinaryEdge(object):
    """Format binary edge."""

    def __init__(
        self,
        selected_label=None,
    ):
        """
        Args:
            select_label Optional(int): if multilabel edges are given,
                the label set with this argument will be used
                (assuming that the edge is indexed from 0)
        """
        self._label = selected_label
        if self._label is not None:
            assert isinstance(self._label, int)

    def __call__(self, results):
        # non-otf
        if "gt_semantic_edge" in results:
            edge = results["gt_semantic_edge"]

            if edge.ndim == 3:
                assert self._label is not None
                # ordered (h, w, trainId)
                edge = edge[:, :, self._label]

            assert edge.ndim == 2
            # No need for hacks to convert dataset
            results["gt_semantic_edge"] = DC(
                to_tensor(edge[None, ...].astype(np.int64)),
                stack=True,
            )
        elif "gt_semantic_seg" in results:
            mask2edge = results.get("mask2edge", None)
            assert mask2edge, "ERR: no mask2edge inside `results`"

            if results.get("inst_sensitive", False):
                inst_map = results.get("gt_inst_seg", None)
                assert inst_map is not None, "ERR: instance map is not available"
                out = mask2edge(
                    mask=results["gt_semantic_seg"],
                    inst_mask=inst_map,
                )
                # remove it from memory?
                del results["gt_inst_seg"]
            else:
                out = mask2edge(mask=results["gt_semantic_seg"])

            edge = out["edge"]

            if edge.ndim == 3:
                assert self._label is not None
                # ordered (trainId, h, w)
                edge = edge[self._label]

            assert edge.ndim == 2
            results["gt_semantic_edge"] = DC(
                to_tensor(edge[None, ...].astype(np.int64)),
                stack=True,
            )

        return results


@PIPELINES.register_module()
class FormatJoint(object):
    """Format multilabel edge and segmentation"""

    def __init__(
        self,
        reduce_zero_label=False,
        reduce_zero_label_edge=False,
    ):
        self.reduce_zero_label = reduce_zero_label
        self.reduce_zero_label_edge = reduce_zero_label_edge

    def __call__(self, results):
        # non-otf
        if "gt_semantic_edge" in results:
            edge = results["gt_semantic_edge"]

            # HACK: decode RGB to 24bit array
            # it's only possible to encode 24 classes
            edge = np.unpackbits(
                edge,
                axis=2,
            )[:, :, -1 : -(results["num_classes"] + 1) : -1]
            edge = np.ascontiguousarray(edge.transpose(2, 0, 1))

            # convert to long
            results["gt_semantic_edge"] = DC(
                to_tensor(edge.astype(np.int64)),
                stack=True,
            )

            # convert segmentation mask as well
            if "gt_semantic_seg" in results:
                # convert to long
                results["gt_semantic_seg"] = DC(
                    to_tensor(results["gt_semantic_seg"][None, ...].astype(np.int64)),
                    stack=True,
                )

        elif "gt_semantic_seg" in results:
            mask2edge = results.get("mask2edge", None)
            assert mask2edge, "ERR: no mask2edge inside `results`"

            if results["inst_sensitive"]:
                inst_map = results.get("gt_inst_seg", None)
                assert inst_map is not None, "ERR: instance map is not available"
                out = mask2edge(
                    mask=results["gt_semantic_seg"],
                    inst_mask=inst_map,
                )
                # remove it from memory?
                del results["gt_inst_seg"]
            else:
                out = mask2edge(mask=results["gt_semantic_seg"])

            mask = out["mask"]
            if self.reduce_zero_label:
                # avoid using underflow conversion
                mask[mask == 0] = 255
                mask = mask - 1
                mask[mask == 254] = 255

            # out is a dict('mask'=..., 'edge'=...)
            results["gt_semantic_seg"] = DC(
                to_tensor(mask[None, ...].astype(np.int64)),
                stack=True,
            )

            # edge
            edge = out["edge"]
            if self.reduce_zero_label_edge:
                # remove zero index
                edge = edge[1:]

            results["gt_semantic_edge"] = DC(
                to_tensor(edge.astype(np.int64)),
                stack=True,
            )

        return results


@PIPELINES.register_module()
class FormatJointBinaryEdge:
    """Format binary edge and segmentation.

    Args:
        select_label Optional(int): if multilabel edges are given,
            the label set with this argument will be used
            (assuming that the edge is indexed from 0)
    """

    def __init__(
        self,
        selected_label=None,
    ):
        self._label = selected_label
        if self._label is not None:
            assert isinstance(self._label, int)

    def __call__(self, results):
        # non-otf
        if "gt_semantic_edge" in results:
            edge = results["gt_semantic_edge"]

            if edge.ndim == 3:
                assert self._label is not None
                # ordered (h, w, trainId)
                edge = edge[:, :, self._label]

            assert edge.ndim == 2
            # No need for hacks to convert dataset
            results["gt_semantic_edge"] = DC(
                to_tensor(edge[None, ...].astype(np.int64)),
                stack=True,
            )

            # convert segmentation mask as well
            if "gt_semantic_seg" in results:
                # convert to long
                results["gt_semantic_seg"] = DC(
                    to_tensor(results["gt_semantic_seg"][None, ...].astype(np.int64)),
                    stack=True,
                )
        elif "gt_semantic_seg" in results:
            mask2edge = results.get("mask2edge", None)
            assert mask2edge, "ERR: no mask2edge inside `results`"

            if results.get("inst_sensitive", False):
                inst_map = results.get("gt_inst_seg", None)
                assert inst_map is not None, "ERR: instance map is not available"
                out = mask2edge(
                    mask=results["gt_semantic_seg"],
                    inst_mask=inst_map,
                )
                # remove it from memory?
                del results["gt_inst_seg"]
            else:
                out = mask2edge(mask=results["gt_semantic_seg"])

            edge = out["edge"]

            if edge.ndim == 3:
                assert self._label is not None
                # ordered (trainId, h, w)
                edge = edge[self._label]

            assert edge.ndim == 2

            results["gt_semantic_edge"] = DC(
                to_tensor(edge[None, ...].astype(np.int64)),
                stack=True,
            )
            results["gt_semantic_seg"] = DC(
                to_tensor(out["mask"][None, ...].astype(np.int64)),
                stack=True,
            )

        return results


@PIPELINES.register_module()
class JointFormatBundle(object):
    def __init__(self, **kwargs):
        self.format_image = FormatImage()
        self.format_joint = FormatJoint(**kwargs)

    def __call__(self, results):
        results = self.format_image(results)
        results = self.format_joint(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class JointBinaryFormatBundle(object):
    def __init__(self, selected_label=None):
        self.format_image = FormatImage()
        self.format_joint = FormatJointBinaryEdge(selected_label=selected_label)

    def __call__(self, results):
        results = self.format_image(results)
        results = self.format_joint(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


# avoid naming `DefaultFormatBundle` since we want to use the original
@PIPELINES.register_module(force=True)
class EdgeFormatBundle(object):
    def __init__(self):
        self.format_image = FormatImage()
        self.format_edge = FormatEdge()

    def __call__(self, results):
        # chain together formatting
        results = self.format_image(results)
        results = self.format_edge(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module(force=True)
class BinaryEdgeFormatBundle(object):
    def __init__(self, selected_label=None):
        self.format_image = FormatImage()
        self.format_edge = FormatBinaryEdge(selected_label=selected_label)

    def __call__(self, results):
        # chain together formatting
        results = self.format_image(results)
        results = self.format_edge(results)
        return results

    def __repr__(self):
        return self.__class__.__name__
