_base_ = [
    "../_base_/models/deeplabv3plus_r50b-d8.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
model = dict(
    backbone=dict(
        stem_stride_size=1,
        dilations=(2, 2, 2, 4),
        strides=(1, 2, 2, 1),
    ),
    auxiliary_head=None,
)
