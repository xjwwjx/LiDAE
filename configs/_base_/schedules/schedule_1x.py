# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optim_wrapper = dict(
#     clip_grad=dict(max_norm=35, norm_type=2),
#     optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
#     paramwise_cfg=dict(bias_decay_mult=0.0, bias_lr_mult=2.0),
#     type='OptimWrapper')
#
# param_scheduler = [
#     dict(
#         begin=0,
#         by_epoch=False,
#         end=500,
#         factor=0.3333333333333333,
#         type='ConstantLR'),
#     dict(
#         begin=0,
#         by_epoch=True,
#         end=12,
#         gamma=0.1,
#         milestones=[
#             8,
#             11,
#         ],
#         type='MultiStepLR'),
# ]

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=450),
    dict(
        type='MultiStepLR',
        begin=0,
        end=15,
        by_epoch=True,
        milestones=[8, 11, 14],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=15, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0.0, bias_lr_mult=2.0),
    type='OptimWrapper')

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
#
# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)


# # Default setting for scaling LR automatically
# param_scheduler = [
#     dict(
#         begin=0, by_epoch=False, end=450, start_factor=0.001, type='LinearLR'),
#     dict(
#         begin=0,
#         by_epoch=True,
#         end=12,
#         gamma=0.1,
#         milestones=[
#             8,
#             11,
#         ],
#         type='MultiStepLR'),
# ]
