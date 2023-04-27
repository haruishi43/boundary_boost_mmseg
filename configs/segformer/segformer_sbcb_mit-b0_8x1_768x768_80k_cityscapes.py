_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/cityscapes_768x768.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (768, 768)
train_pipeline_otf = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
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
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"  # noqa
model = dict(
    type="SBCBEncoderDecoder",
    pass_input_image=True,
    backbone=dict(
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type="Pretrained",
            checkpoint=checkpoint,
        ),
    ),
    auxiliary_edge_head=dict(
        type="AuxCASENetHead",
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_multilabel=dict(type="MultiLabelEdgeLoss", loss_weight=5.0),
    ),
    test_cfg=dict(mode="slide", crop_size=(768, 768), stride=(512, 512)),
)

# optimizer
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
