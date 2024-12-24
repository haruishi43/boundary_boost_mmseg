_base_ = [
    "../_base_/models/pspnet_hed-r50-d8.py",
    "../_base_/datasets/cityscapes_joint_769x769.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
crop_size = (769, 769)
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegEdgeDataPreProcessor",
    size=crop_size,
)
model = dict(
    type="SegEdgeEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pass_input_image=True,
    backbone=dict(
        return_stem=True,
    ),
    decode_head=dict(
        align_corners=True,
        in_index=4,
    ),
    auxiliary_head=dict(
        align_corners=True,
        in_index=3,
    ),
    edge_decode_head=dict(
        type="CASENetHead",
        in_channels=[64, 256, 512, 2048],
        in_index=[0, 1, 2, 4],
        pass_input_image=True,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        pred_key="fuse",
        log_keys=("fuse", "last"),
        loss_decode=dict(
            mlbl=dict(
                fuse=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
                last=dict(type="MultiLabelEdgeLoss", loss_weight=1.0),
            ),
        ),
    ),
    test_cfg=dict(mode="slide", crop_size=(769, 769), stride=(513, 513)),
)
