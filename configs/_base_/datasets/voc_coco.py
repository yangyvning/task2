# configs/_base_/datasets/voc_coco.py

# 直接继承 COCO 格式训练配置
dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'

# 20 个 VOC 类
classes = (
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
    'cow','diningtable','dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc_2007_trainval.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/voc_2007_test.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        test_mode=True,
        metainfo=dict(classes=classes),
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', metric=['bbox', 'segm'])
test_evaluator = val_evaluator
