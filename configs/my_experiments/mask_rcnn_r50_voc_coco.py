# configs/my_experiments/mask_rcnn_r50_voc_coco.py


_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'

# VOC 20 类别
classes = (
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
)

# --------------------- Data Pipelines ---------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    dict(type='RandomCrop', crop_size=(0.8, 0.8), allow_negative_crop=True),
    dict(type='PhotoMetricDistortion')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

# ------------------- Dataloaders Setup --------------------

train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/voc07_trainval.json',   # 相对于 data_root
        data_prefix=dict(img=''),

        metainfo=dict(classes=classes),
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/voc07_test.json',

        data_prefix=dict(img=''),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
test_dataloader = val_dataloader

# ------------------- Model Adjustment --------------------


model = dict(roi_head=dict(bbox_head=dict(num_classes=20),
                           mask_head=dict(num_classes=20)))
# ------------------- Evaluators --------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/voc07_test.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator
log_config = dict(
    interval=50,  # 每50个迭代记录一次日志
    hooks=[
        dict(type='TextLoggerHook'),  # 保留文本日志
        dict(type='TensorboardLoggerHook',  # 启用TensorBoard
             log_dir='work_dirs/tensorboard',  # 日志保存路径
             interval=50,  # 与文本日志同步
             reset_flag=False,
             by_epoch=True)  # 按轮次记录
    ])
