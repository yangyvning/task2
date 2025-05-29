
import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config

# ------------------------ 配置参数 --------------------------
# Mask R-CNN配置文件和权重路径
mask_rcnn_config = 'configs/my_experiments/mask_rcnn_r50_voc_coco.py'
mask_rcnn_checkpoint = 'work_dirs/mask_rcnn_r50_voc_coco/epoch_12.pth'

# Sparse R-CNN配置文件和权重路径
sparse_rcnn_config = 'configs/my_experiments/sparse_rcnn_r50_voc_coco.py'
sparse_rcnn_checkpoint = 'work_dirs/sparse_rcnn_r50_voc_coco/epoch_24.pth'

# 测试图像路径（非VOC数据集）
test_img_dir = 'data/VOCdevkit'  # 自定义图像目录
img_list = ['t1.jpg', 't2.jpg', 't3.jpg']  # 三张包含VOC类别但不在VOC中的图像

# 输出可视化目录
output_dir = 'work_dirs/visualization/'
os.makedirs(output_dir, exist_ok=True)

# ------------------------ 初始化模型 --------------------------
# 初始化Mask R-CNN模型
mask_rcnn_model = init_detector(mask_rcnn_config, mask_rcnn_checkpoint, device='cuda:0')

# 初始化Sparse R-CNN模型（支持实例分割）
sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')

# ------------------------ 可视化工具设置 --------------------------
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = mask_rcnn_model.dataset_meta  # 继承类别信息


# ------------------------ 可视化函数 --------------------------
def visualize_results(model, img_path, model_type='mask_rcnn'):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)

    # 绘制结果（边界框、掩码、标签、得分）
    visualizer.set_image(img.copy())
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.5,  # 显示得分阈值
        show=False
    )
    return visualizer.get_image()


# ------------------------ 生成对比图像 --------------------------
for img_name in img_list:
    img_path = os.path.join(test_img_dir, img_name)

    # 推理并可视化
    mask_img = visualize_results(mask_rcnn_model, img_path, model_type='mask_rcnn')
    sparse_img = visualize_results(sparse_rcnn_model, img_path, model_type='sparse_rcnn')

    # 横向拼接对比图
    combined = np.concatenate([mask_img, sparse_img], axis=1)

    # 保存结果
    output_path = os.path.join(output_dir, f'compare_{img_name}')
    plt.imsave(output_path, combined)
    print(f'结果已保存至：{output_path}')