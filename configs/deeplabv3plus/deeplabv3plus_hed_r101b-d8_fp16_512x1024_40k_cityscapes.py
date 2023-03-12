_base_ = "./deeplabv3plus_hed_r101b-d8_512x1024_40k_cityscapes.py"
# fp16 settings
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
# fp16 placeholder
fp16 = dict()
