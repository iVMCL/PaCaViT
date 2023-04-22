# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
model = dict(
    type="PaCaEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(
        type="PaCaSegHead",
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            avg_non_ignore=True,
            use_sigmoid=False,
            loss_weight=1.0,
        ),
        aux_loss_decode=dict(
            type="CrossEntropyLoss",
            avg_non_ignore=True,
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
