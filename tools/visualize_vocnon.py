import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config

# --------------- 设置模型配置和权重路径 -------------------
mask_rcnn_config = 'configs/my_experiments/mask_rcnn_r50_voc_coco.py'
mask_rcnn_checkpoint = 'work_dirs/mask_rcnn_r50_voc_coco/epoch_12.pth'

sparse_rcnn_config = 'configs/my_experiments/sparse_rcnn_r50_voc_coco.py'
sparse_rcnn_checkpoint = 'work_dirs/sparse_rcnn_r50_voc_coco/epoch_24.pth'

# 自定义图像路径列表（非VOC数据集）
img_list = [
    'data/VOCdevkit/t1.jpg',
    'data/VOCdevkit/t2.jpg',
    'data/VOCdevkit/t3.jpg',
]

# 输出可视化结果目录
output_dir = 'work_dirs/visualization_nonvoc/'
os.makedirs(output_dir, exist_ok=True)

# --------------- 初始化模型 -------------------
mask_rcnn_model = init_detector(Config.fromfile(mask_rcnn_config), mask_rcnn_checkpoint, device='cuda:0')
sparse_rcnn_model = init_detector(Config.fromfile(sparse_rcnn_config), sparse_rcnn_checkpoint, device='cuda:0')

# --------------- 通用可视化函数 -------------------
def visualize_model_result(model, img_path, model_name, palette_color):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)

    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta
    visualizer.dataset_meta['palette'] = [palette_color] * len(model.dataset_meta['classes'])

    visualizer.set_image(img)
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.3,
        show=False
    )

    return visualizer.get_image()

# --------------- 可视化并保存 -------------------
for img_path in img_list:
    basename = os.path.basename(img_path)

    # Mask R-CNN 检测（红色调）
    mask_result_img = visualize_model_result(mask_rcnn_model, img_path, 'Mask R-CNN', palette_color=(255, 0, 0))

    # Sparse R-CNN 检测（蓝色调）
    sparse_result_img = visualize_model_result(sparse_rcnn_model, img_path, 'Sparse R-CNN', palette_color=(0, 0, 255))

    # 横向拼接
    combined = np.concatenate([mask_result_img, sparse_result_img], axis=1)

    # 保存
    save_path = os.path.join(output_dir, f'compare_{basename}')
    plt.imsave(save_path, combined)
    print(f'图像 {basename} 的对比结果已保存至: {save_path}')
