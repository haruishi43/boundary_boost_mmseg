# model settings
_base_ = "./pspnet_r50-d8.py"
model = dict(
    # pretrained="torchvision://resnet50",
    backbone=dict(
        type="HEDResNetV1c",
        # type="HEDResNet",  # don't use v1c
        stem_stride_size=1,
        # return_stem=True,
        dilations=(2, 2, 2, 4),
        strides=(1, 2, 2, 1),
    ),
)
