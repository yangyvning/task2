import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config


# ------------------------ 配置参数 --------------------------
mask_rcnn_config = 'configs/my_experiments/mask_rcnn_r50_voc_coco.py'
mask_rcnn_checkpoint = 'work_dirs/mask_rcnn_r50_voc_coco/epoch_12.pth'

sparse_rcnn_config = 'configs/my_experiments/sparse_rcnn_r50_voc_coco.py'
sparse_rcnn_checkpoint = 'work_dirs/sparse_rcnn_r50_voc_coco/epoch_1.pth'

test_img_dir = r'data\VOCdevkit\VOC2007\JPEGImages'
img_list = ['000001.jpg', '000002.jpg', '000003.jpg', '000007.jpg']

output_dir = 'work_dirs/visualization/'
os.makedirs(output_dir, exist_ok=True)


# ------------------------ 修改Mask R-CNN配置以获取RPN提案 --------------------------
def modify_config_for_proposals(config_path):
    cfg = Config.fromfile(config_path)
    if hasattr(cfg.model.test_cfg, 'rpn') and cfg.model.test_cfg.rpn is not None:
        cfg.model.test_cfg.rpn.keep_all_stages = True
    if hasattr(cfg.model.test_cfg, 'rcnn') and cfg.model.test_cfg.rcnn is not None:
        cfg.model.test_cfg.rcnn.score_thr = 0.3
    return cfg


# ------------------------ bbox和mask裁剪函数 --------------------------
def clip_bboxes_masks(result, img_shape):
    h, w = img_shape[:2]
    if hasattr(result, 'pred_instances'):
        # 先处理bboxes，clip到图像范围
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
        result.pred_instances.bboxes = bboxes

        # 如果有mask，且是BitmapMask，mmdet默认已处理越界
        # 但如果mask是Polygon格式，需遍历每个点clip
        if hasattr(result.pred_instances, 'masks'):
            masks = result.pred_instances.masks
            # 这里只示范BitmapMask，不同版本的mmdet可能要做适配
            # 若为Polygon，可以根据需要做clip，这里不详细写了
    return result


# ------------------------ 初始化模型 --------------------------
mask_rcnn_cfg = modify_config_for_proposals(mask_rcnn_config)
mask_rcnn_model = init_detector(mask_rcnn_cfg, mask_rcnn_checkpoint, device='cuda:0')
sparse_rcnn_model = init_detector(sparse_rcnn_config, sparse_rcnn_checkpoint, device='cuda:0')

visualizer = DetLocalVisualizer()
visualizer.dataset_meta = mask_rcnn_model.dataset_meta


# ------------------------ 可视化函数 --------------------------
def visualize_results(model, img_path, model_type='mask_rcnn', show_proposals=False):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)

    # 只对sparse_rcnn的结果做裁剪，防止越界
    if model_type == 'sparse_rcnn':
        result = clip_bboxes_masks(result, img.shape)

    visualizer.set_image(img)

    # 如果需要画RPN proposals，mask_rcnn有，sparse_rcnn通常没有
    if show_proposals and hasattr(result, 'rpn_proposals'):
        proposals = result.rpn_proposals[0][:50]
        proposals[:, 0::2] = np.clip(proposals[:, 0::2], 0, img.shape[1] - 1)
        proposals[:, 1::2] = np.clip(proposals[:, 1::2], 0, img.shape[0] - 1)
        visualizer.draw_bboxes(proposals, edge_colors='green', line_widths=1, alpha=0.3)

    if model_type == 'mask_rcnn':
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=0.3,
            show=False
        )
    elif model_type == 'sparse_rcnn':
        # 用蓝色画Sparse R-CNN结果
        visualizer.dataset_meta['palette'] = [(0, 0, 255)] * 20
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=0.3,
            show=False
        )
        # 恢复默认palette
        visualizer.dataset_meta['palette'] = mask_rcnn_model.dataset_meta['palette']

    return visualizer.get_image()


# ------------------------ 生成对比图像 --------------------------
for img_name in img_list:
    img_path = os.path.join(test_img_dir, img_name)

    mask_img = visualize_results(mask_rcnn_model, img_path, show_proposals=True)
    mask_final = visualize_results(mask_rcnn_model, img_path, show_proposals=False)
    sparse_img = visualize_results(sparse_rcnn_model, img_path, model_type='sparse_rcnn')

    combined = np.concatenate([mask_img, mask_final, sparse_img], axis=1)

    output_path = os.path.join(output_dir, f'compare_{img_name}')
    plt.imsave(output_path, combined)
    print(f'可视化结果已保存至：{output_path}')
