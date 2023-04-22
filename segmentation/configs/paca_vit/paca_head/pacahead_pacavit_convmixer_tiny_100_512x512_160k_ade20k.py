_base_ = [
    "../../_base_/models/paca_head.py",
    "../../_base_/datasets/ade20k.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k.py",
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="pacavit_convmixer_tiny_100_downstream",
        drop_path_rate=0.1,
        layer_scale=None,
        pretrained=(
            "../work_dirs/classification/cvpr23_paca/IMNET_224_pacavit_convmixer_tiny_100.pth"
        ),
        downstream_cluster_num=[200, 200, 200, 200],
    ),
    decode_head=dict(in_channels=[96, 192, 320, 384], num_classes=150),
)


# AdamW optimizer
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "clustering.4": dict(lr_mult=10.0),  # .4 for p2cconv
            "head": dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    ),
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
