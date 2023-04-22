_base_ = [
    "../../_base_/models/mask-rcnn_r50_fpn.py",
    "../../_base_/datasets/coco_instance.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py",
]


model = dict(
    backbone=dict(
        _delete_=True,
        type="pacavit_small_p2cconv_2_0_downstream",
        drop_path_rate=0.1,
        layer_scale=None,
        pretrained=(
            "../work_dirs/classification/cvpr23_paca/IMNET_224_pacavit_small_p2cconv_2_0.pth"
        ),
    ),
    neck=dict(in_channels=[96, 192, 320, 384]),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile", file_client_args={{_base_.file_client_args}}),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
        }
    ),
    optimizer=dict(
        _delete_=True, type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05
    ),
)
