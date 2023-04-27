_base_ = ["./segformer_sbcb_mit-b0_8x1_768x768_80k_cityscapes.py"]

model = dict(
    backbone=dict(
        strides=[2, 2, 2, 2],  # change reduction from [4, 2, 2, 2]
    ),
)
