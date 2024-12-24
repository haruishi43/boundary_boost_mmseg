# dataset settings
dataset_type = "CityscapesDataset"
data_root = "data/cityscapes/"

crop_size = (512, 1024)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="LoadMultiLabelEdge"),  # Load mlbl edge
    dict(
        type="RandomResize",
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="DecodeMultiLabelEdge", num_classes=19),  # decode channels to mlbl
    dict(type="PackSegEdgeInputs"),
]
train_otf_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadOTFSegAnnotations"),  # also loads inst map
    dict(
        type="RandomResize",
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="OTFSeg2MultiLabelEdge", radius=2),  # convert seg map to mlbl edges
    dict(type="FormatSegMask"),  # format seg mask to train ids
    dict(type="PackSegEdgeInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    # dict(type="LoadMultiLabelEdge"),
    # dict(type="DecodeMultiLabelEdge", num_classes=19),  # decode channels to mlbl
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            # [dict(type="LoadMultiLabelEdge")],
            # [dict(type="DecodeMultiLabelEdge", num_classes=19)],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    # preprocessed
    # dataset=dict(
    #     type="CityscapesSegEdgeDataset",
    #     data_root=data_root,
    #     data_prefix=dict(
    #         img_path="leftImg8bit/train",
    #         seg_map_path="gtFine/train",
    #         mlbl_edge_path="gtBlette/train",
    #     ),
    #     inst_sensitive=True,
    #     thin=False,
    #     edge_mode="mlbl",
    #     pipeline=train_pipeline,
    # ),
    # otf
    dataset=dict(
        type="CityscapesOTFEdgeDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="leftImg8bit/train",
            seg_map_path="gtFine/train",
        ),
        inst_sensitive=True,
        pipeline=train_otf_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="leftImg8bit/val", seg_map_path="gtFine/val"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
