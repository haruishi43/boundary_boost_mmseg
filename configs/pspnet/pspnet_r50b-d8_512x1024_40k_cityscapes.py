_base_ = [
    "../_base_/models/pspnet_r50b-d8.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

model = dict(auxiliary_head=None)
