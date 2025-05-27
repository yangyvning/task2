# configs/_base_/datasets/voc_instance.py

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

metainfo = {
    'classes': (
        'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
        'cow','diningtable','dog','horse','motorbike','person','pottedplant',
        'sheep','sofa','train','tvmonitor'
    ),
    'palette': [
        (106, 0, 228), (0, 0, 255), (0, 128, 255), (0, 255, 255),
        (0, 255, 0), (255, 128, 0), (255, 255, 0), (255, 0, 0),
        (255, 0, 255), (128, 0, 255), (0, 0, 128), (0, 255, 128),
        (128, 128, 0), (128, 0, 0), (128, 64, 0), (0, 128, 0),
        (128, 0, 128), (0, 128, 128), (64, 0, 128), (64, 128, 128)
    ]
}

# 训练/测试共用的 pipeline，保证 mask 字段一定被加载
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/trainval.txt',
        data_prefix=dict(
            img_path='VOC2007/JPEGImages/',
            ann_path='VOC2007/Annotations/'
        ),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/val.txt',
        data_prefix=dict(
            img_path='VOC2007/JPEGImages/',
            ann_path='VOC2007/Annotations/'
        ),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
