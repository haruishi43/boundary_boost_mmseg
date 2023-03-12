_base_ = [
    "../_base_/models/deeplabv3_r50b-d8.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
model = dict(
    pretrained="torchvision://resnet101",
    backbone=dict(type="ResNet", depth=101),
    auxiliary_head=None,
)
