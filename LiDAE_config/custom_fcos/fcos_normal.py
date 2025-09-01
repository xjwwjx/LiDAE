
_base_ = [
    # '../_base_/datasets/hazydet_det.py',
    '../_base_/datasets/visdrone.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

INF = 1e8
# model settings
model = dict(
    type='FCOS',

    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),

    dehazing_network=dict(
        type='IA',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mmdet/models/dehaze_module/IA.pth')
    ),


    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),

    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs='on_output',
    #     num_outs=4,
    #     relu_before_extra_convs=True),

    neck=dict(
        type='Custom_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=4,
        num_fe=2,
    ),


    bbox_head=dict(
        type='Custom_Head',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        strides=[8, 16, 32, 64],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, INF)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),

    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# # learning rate
# param_scheduler = [
#     dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]
#
#
# # optimizer
# optim_wrapper = dict(
#     clip_grad=dict(max_norm=35, norm_type=2),
#     optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
#     paramwise_cfg=dict(bias_decay_mult=0.0, bias_lr_mult=2.0),
#     type='OptimWrapper')

