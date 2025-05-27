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
mask_rcnn_checkpoint = 'work_dirs/mask_rcnn_r50_voc_coco/epoch_12.pth'  # 假设训练好的权重路径

# Sparse R-CNN配置文件和权重路径（需确保版本兼容）
sparse_rcnn_config = 'configs/my_experiments/sparse_rcnn_r50_voc_coco.py'
sparse_rcnn_checkpoint ='work_dirs\sparse_rcnn_r50_voc_coco/r50_300pro_3x_model.pth'

# 测试图像路径（从VOC测试集选取4张）
test_img_dir = 'data\VOCdevkit\VOC2007\JPEGImages'
img_list = ['000001.jpg', '000002.jpg', '000003.jpg', '000007.jpg']

img_list = ['t1.jpg', 't2.jpg', 't3.jpg']
test_img_dir = 'data\VOCdevkit'
# 输出可视化目录
output_dir = 'work_dirs/visualization/'
os.makedirs(output_dir, exist_ok=True)


# ------------------------ 修改Mask R-CNN配置以获取RPN提案 --------------------------
def modify_config_for_proposals(config_path):
    cfg = Config.fromfile(config_path)
    # 修改测试配置以保留RPN提案
    cfg.model.test_cfg.rpn.keep_all_stages = True  # 保留中间结果
    cfg.model.test_cfg.rcnn.score_thr = 0.3  # 降低阈值以显示更多结果
    return cfg


# ------------------------ 初始化模型 --------------------------
# 初始化Mask R-CNN模型（包含提案生成）
mask_rcnn_cfg = modify_config_for_proposals(mask_rcnn_config)
mask_rcnn_model = init_detector(mask_rcnn_cfg, mask_rcnn_checkpoint, device='cuda:0')

# 初始化Sparse R-CNN模型
sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')

# ------------------------ 可视化工具设置 --------------------------
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = mask_rcnn_model.dataset_meta  # 继承类别信息


# ------------------------ 可视化函数 --------------------------
def visualize_results(model, img_path, model_type='mask_rcnn', show_proposals=False):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)

    # 绘制不同结果
    visualizer.set_image(img)
    if show_proposals and 'rpn_proposals' in result:
        # 绘制RPN提案（绿色框）
        proposals = result.rpn_proposals[0][:50]  # 取前50个提案
        visualizer.draw_bboxes(proposals, edge_colors='green', line_widths=1, alpha=0.3)

    if model_type == 'mask_rcnn':
        # 绘制最终检测结果（红色框和掩码）
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=0.3,
            show=False
        )
    elif model_type == 'sparse_rcnn':
        # Sparse R-CNN结果（蓝色框）
        visualizer.dataset_meta['palette'] = [(0, 0, 255)] * 20  # 假设20类，全部设为蓝色
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=0.3,
            show=False
        )
        visualizer.dataset_meta['palette'] = mask_rcnn_model.dataset_meta['palette']  # 恢复默认颜色
    return visualizer.get_image()


# ------------------------ 生成对比图像 --------------------------
for img_name in img_list:
    img_path = os.path.join(test_img_dir, img_name)

    # Mask R-CNN提案与最终结果
    mask_img = visualize_results(mask_rcnn_model, img_path, show_proposals=True)
    # Mask R-CNN最终结果单独
    mask_final = visualize_results(mask_rcnn_model, img_path, show_proposals=False)
    # Sparse R-CNN结果
sparse_img = visualize_results(sparse_rcnn_model, img_path, model_type='sparse_rcnn')

    # 横向拼接对比
combined = np.concatenate([mask_img, mask_final, sparse_img], axis=1)

    # 保存结果
output_path = os.path.join(output_dir, f'compare_{img_name}')
plt.imsave(output_path, combined)
print(f'可视化结果已保存至：{output_path}')
