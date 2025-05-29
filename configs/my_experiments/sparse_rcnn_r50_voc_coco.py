# configs/my_experiments/sparse_rcnn_r50_voc_coco.py

_base_ = [
    '../_base_/models/sparse-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'

classes = (
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000,600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1000,600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2, num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/voc07_trainval.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline
    )
)


val_dataloader = dict(
    batch_size=1, num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/voc07_test.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

model = dict(
    roi_head=dict(
        bbox_head=[dict(type='DIIHead', num_classes=20) for _ in range(6)],
        mask_head=dict(
            type='DynamicMaskHead',
            num_classes=20,
            roi_feat_size=14,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        )
    )
)


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/voc07_test.json',
    metric=['bbox', 'segm']
)

test_evaluator = val_evaluator

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook',
             log_dir='work_dirs/tensorboard',
             interval=500,
             reset_flag=False,
             by_epoch=True)
    ])
