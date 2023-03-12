_base_ = [
    "../_base_/models/deeplabv3_r50b-d8.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 1024)
train_pipeline_otf = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(
        type="Pad",
        size=crop_size,
        pad_val=0,
        seg_pad_val=255,
        edge_pad_val=0,
        inst_pad_val=0,
    ),
    dict(type="JointFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg", "gt_semantic_edge"]),
]

inst_sensitive = True
data = dict(
    train=dict(
        type="OTFJointCityscapesDataset",
        pipeline=train_pipeline_otf,
        img_dir="leftImg8bit/train",
        ann_dir="gtFine/train",
        inst_dir="gtFine/train",
        inst_sensitive=inst_sensitive,
        radius=2,
    ),
)

norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="SBCBEncoderDecoder",
    pass_input_image=True,
    backbone=dict(
        out_indices=(0, 1, 2, 3),
        return_stem=True,
    ),
    decode_head=dict(
        in_index=4,
    ),
    auxiliary_head=None,
    # auxiliary_head=dict(in_index=3),
    auxiliary_edge_head=dict(
        type="AuxCASENetHead",
        in_channels=[64, 256, 512, 2048],
        in_index=[0, 1, 2, 4],
        channels=512,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_multilabel=dict(type="MultiLabelEdgeLoss", loss_weight=5.0),
    ),
)
