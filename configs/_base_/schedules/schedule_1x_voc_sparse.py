# configs/_base_/schedules/schedule_1x_voc_sparse.py
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[9, 11], gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

auto_scale_lr = dict(enable=False, base_batch_size=16)
