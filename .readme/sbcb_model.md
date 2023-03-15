# How to apply SBCB for your own model


There are three steps to applying SBCB to your own model for `mmseg`:
1. Check if the model is supported
2. Choose or make your own auxiliary SBD head
3. Edit the config files

## 1. Check if the model is supported

Most segmentation models consists of a backbone and a head.
The backbone has various stages, and the head usually uses the features from the last stage.
We have to be cautious when applying SBCB to a model, the features being used in the head might be hard-coded.

For example, DeepLabV3+ uses features from the first stage (C1) of the backbone, but we would need to change this since we need to get the features from the stem which is not officially supported by `mmseg`.
We modded the `sep_aspp_head.py` to support this, but it is a simple modification where we change the index of the features for `c1_output`.

In most cases, like PSPNet and FCN` we would not need to modify the decode head.

## 2. Choose or make your own auxiliary SBD head

We provide 3 auxiliary SBD heads for you to choose from:
- `AuxCASENetHead`
- `AuxDFFHead`
- `AuxDDSHead`

You can also make your own auxiliary SBD head by inheriting from `BaseMultiSupervisionHead`.

## 3. Edit the config files

Since we need to use the OTF semantic boundary GT generation pipelines, we need to add the following to the config files:

```python
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
```

Make sure to change parameters for your usecases (like the `crop_size`).

We also need to modify the `model` section of the config file to use the auxiliary SBD head.

```python
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
    auxiliary_edge_head=dict(
        type="AuxCASENetHead",
        in_channels=[64, 256, 512, 2048],
        in_index=[0, 1, 2, 4],
        channels=512,
        num_classes=19,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_multilabel=dict(type="MultiLabelEdgeLoss", loss_weight=5.0),
    ),
)
```

Couple things to note:
- We use `SBCBEncoderDecoder` as the `type` for the `model` section. This is a subclass of the `EncoderDecoder` model in `mmseg` which adds two main functionality:
    - It allows us to pass the input image to the auxiliary SBD head.
    - It allows us to use the auxiliary SBD head (`auxiliary_edge_head`).
- We set `pass_input_image` to `True` to pass the input image to the auxiliary SBD head. The input image is added to the end of the list of backbone features.
- We set `return_stem` to `True` to return the stem of the backbone. This is needed for the auxiliary SBD head when using ResNet as the backbone. Some other backbones that we provide, like HRNet, also supports `return_stem` argument. The stem features are prepended to the first of the list of backbone features.
- We need to modify the `in_index` of the `decode_head` to account for the added `stem` features.
- We also need to add the `auxiliary_edge_head` section to the `model` section. This is where we specify the auxiliary SBD head to use. We also need add the `in_channels` and `in_index` (remember to account for the `stem` for some of the backbones).
    - For example, in this auxiliary edge head, we use Stem, Stage1, Stage2, and Stage4 of the ResNet backbone.
