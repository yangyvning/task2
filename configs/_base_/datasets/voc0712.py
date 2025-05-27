# configs/_base_/datasets/voc0712.py

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

backend_args = None

# 图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # BGR
    std=[58.395, 57.12, 57.375],
    to_rgb=False
)

# 训练数据预处理流水线，加载掩膜
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 验证/测试数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# VOC2007 & VOC2012 训练集拼接
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['dataset_type'],
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(
                    sub_data_root='VOC2007/',
                    seg_prefix='VOC2007/SegmentationObject'
                ),
                filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
            ),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(
                    sub_data_root='VOC2012/',
                    seg_prefix='VOC2012/SegmentationObject'
                ),
                filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args
            )
        ]
    )
)

# 验证集
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(
            sub_data_root='VOC2007/',
            seg_prefix='VOC2007/SegmentationObject'
        ),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

# 评估器，只评估检测和分割mAP@0.5
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
