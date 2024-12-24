#!/usr/bin/env python3

import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmseg.models.utils import resize
from mmseg.structures import build_pixel_sampler

from mmseg.utils import ConfigType, SampleList
from .utils import init_edge_loss
from ..losses import calc_edge_metrics


DEFAULT_PRED_KEY = "pred"


class BaseEdgeDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for Decode Heads.

    Based on mmseg's BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict edge results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict edge results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): Input channels.
        num_classes (int): Number of classes.
        threshold (float): Threshold for binary edge in the case of
            `num_classes==1`. This argument is also used for logging acc.
            Default: None.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        pass_input_image (bool): whether to pass the input image
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        pred_key (str): The key to specify the output logit.
            Default: DEFAULT_PRED_KEY
        log_keys (Sequence[str]): The keys to specify the logging logits.
            This is to reduce computation on logits that don't require logging.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
            e.g. dict(type='CrossEntropyLoss'),
                [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
                dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of edge map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    BIN_KEYS = None
    MLBL_KEYS = None

    def __init__(
        self,
        in_channels,
        *,  # requires keyword arguments
        num_classes,
        threshold=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        in_index=-1,
        pass_input_image=False,
        input_transform=None,
        pred_key=DEFAULT_PRED_KEY,
        log_keys=(DEFAULT_PRED_KEY,),
        loss_decode=dict(
            mlbl=dict(
                pred=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
            ),
        ),
        no_accuracy: bool = False,
        ignore_index=255,
        sampler=None,
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01),
    ) -> None:
        super().__init__(init_cfg)

        self._init_inputs(in_channels, in_index, input_transform)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.pass_input_image = pass_input_image

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if num_classes == 2:
            warnings.warn(
                "For binary edges, we suggest using"
                "`num_classes = 1` to define the output"
                "channels of detector, and use `threshold`"
                "to convert `edge_logits` into a prediction"
                "applying a threshold."
                "We will force `num_classes = 1`."
            )
            num_classes = 1

        if threshold is None:
            warnings.warn("threshold is not defined, and defaults" "to 0.5")
            threshold = 0.5

        self.num_classes = num_classes
        self.threshold = threshold
        self.no_accuracy = no_accuracy

        # initialize loss
        binary_loss, binary_keys, mlbl_loss, mlbl_keys = init_edge_loss(
            loss_decode, num_classes
        )
        self.BIN_KEYS = binary_keys
        self.MLBL_KEYS = mlbl_keys
        self.binary_loss = binary_loss
        self.mlbl_loss = mlbl_loss

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        # check for `pred_key`
        assert pred_key is not None
        if self.num_classes > 1:
            assert pred_key in self.MLBL_KEYS
        else:
            assert pred_key in self.BIN_KEYS
        self.pred_key = pred_key

        # set log keys
        if log_keys is None:
            log_keys = [pred_key]
        else:
            assert isinstance(log_keys, (list, tuple))
            for key in log_keys:
                assert key in self.BIN_KEYS + self.MLBL_KEYS
        self.log_keys = log_keys

    def extra_repr(self):
        """Extra repr."""
        s = (
            f"input_transform={self.input_transform}, "
            f"ignore_index={self.ignore_index}, "
            f"align_corners={self.align_corners}"
        )
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                # resize_concat
                self.in_channels = sum(in_channels)
            else:
                # multiple_select
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        # exceptation: a list/tuple of Tensors
        # (stem, layer1, layer2, layer3, layer4, (input_image))

        if self.pass_input_image:
            if isinstance(inputs, tuple):
                # if inputs is a tuple, we need to convert it to a list
                inputs = list(inputs)
            # pop the image located in the last index
            image = inputs.pop()

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        if self.pass_input_image:
            if not isinstance(inputs, list):
                # if inputs is not a list, we need to convert it to a list
                inputs = [inputs]
            inputs.append(image)

        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def loss(
        self,
        inputs: Tuple[Tensor],
        batch_data_samples: SampleList,
        train_cfg: ConfigType,
    ) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`BasicEdgeDataSample`]): The edge
                data samples. It usually includes information such
                as `img_metas` or `gt_bin_edge`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        edge_logits = self.forward(inputs)

        assert isinstance(edge_logits, dict), "edge_logits must be a dict of Tensors"
        losses = self.loss_by_feat(edge_logits, batch_data_samples)

        return losses

    def predict(
        self, inputs: Tuple[Tensor], batch_img_metas: List[dict], test_cfg: ConfigType
    ) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `blette/datasets/pipelines/formatting.py:PackEdgeInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs edge logits map.
        """
        edge_logits = self.forward(inputs)

        assert isinstance(edge_logits, dict), "edge_logits must be a dict of Tensors"
        # select a single logit for output
        edge_logits = edge_logits[self.pred_key]

        return self.predict_by_feat(edge_logits, batch_img_metas)

    def _stack_batch_bin_gt(self, batch_data_samples: SampleList) -> Tensor:
        # FIXME: might need to unsqueeze here
        # but it seems that PackEdgeInputs already does that
        gt_bin_edges = [
            data_sample.gt_bin_edge.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_bin_edges, dim=0)

    def _stack_batch_mlbl_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_mlbl_edges = [
            data_sample.gt_mlbl_edge.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_mlbl_edges, dim=0)

    def loss_by_feat(self, edge_logits: dict, batch_data_samples: SampleList) -> dict:
        """Compute edge loss.

        Args:
            edge_logits (dict[str, Tensor]): The output from decode head forward function.
            batch_data_samples (List[:obj:`BasicEdgeDataSample`]): The edge
                data samples. It usually includes information such
                as `metainfo` and `gt_bin_edge`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        loss = dict()

        # binary edge loss
        for key in self.BIN_KEYS:
            bin_edge_logits = edge_logits[key]
            bin_edge_label = self._stack_batch_bin_gt(batch_data_samples)

            bin_edge_logits = resize(
                input=bin_edge_logits,
                size=bin_edge_label.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

            if self.sampler is not None:
                edge_weight = self.sampler.sample(bin_edge_logits, bin_edge_label)
            else:
                edge_weight = None

            if not isinstance(self.binary_loss[key], nn.ModuleList):
                losses_decode = [self.binary_loss[key]]
            else:
                losses_decode = self.binary_loss[key]
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        bin_edge_logits,
                        bin_edge_label,
                        weight=edge_weight,
                        ignore_index=self.ignore_index,
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        bin_edge_logits,
                        bin_edge_label,
                        weight=edge_weight,
                        ignore_index=self.ignore_index,
                    )

            if (key in self.log_keys) and (not self.no_accuracy):
                for name, v in calc_edge_metrics(
                    bin_edge_logits,
                    bin_edge_label,
                    thresh=self.threshold,
                ).items():
                    loss[name + "_" + key] = v

        # multilabel edge loss
        for key in self.MLBL_KEYS:
            mlbl_edge_logits = edge_logits[key]
            mlbl_edge_label = self._stack_batch_mlbl_gt(batch_data_samples)

            mlbl_edge_logits = resize(
                input=mlbl_edge_logits,
                size=mlbl_edge_label.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

            if self.sampler is not None:
                edge_weight = self.sampler.sample(mlbl_edge_logits, mlbl_edge_label)
            else:
                edge_weight = None

            if not isinstance(self.mlbl_loss[key], nn.ModuleList):
                losses_decode = [self.mlbl_loss[key]]
            else:
                losses_decode = self.mlbl_loss[key]
            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        mlbl_edge_logits,
                        mlbl_edge_label,
                        weight=edge_weight,
                        ignore_index=self.ignore_index,
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        mlbl_edge_logits,
                        mlbl_edge_label,
                        weight=edge_weight,
                        ignore_index=self.ignore_index,
                    )

            if (key in self.log_keys) and (not self.no_accuracy):
                for name, v in calc_edge_metrics(
                    mlbl_edge_logits,
                    mlbl_edge_label,
                    thresh=self.threshold,
                ).items():
                    loss[name + "_" + key] = v

        return loss

    def predict_by_feat(
        self, edge_logits: Tensor, batch_img_metas: List[dict]
    ) -> Tensor:
        """Transform a batch of output edge_logits to the input shape.

        Args:
            edge_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs edge logits map.
        """

        edge_logits = resize(
            input=edge_logits,
            size=batch_img_metas[0]["img_shape"],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return edge_logits
