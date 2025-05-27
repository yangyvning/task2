#（2）四张测试图像上可视化 RPN proposals、Mask R-CNN 最终检测结果 + 对比两种模型的分割与检测
import os
import mmcv
import matplotlib.pyplot as plt
from mmengine import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample

def show_rpn_proposals(model, img_path: str, out_path: str, topk=100):
    """用模型的 RPN head 生成 proposals 并可视化前 topk 框。"""
    img = mmcv.imread(img_path)
    # 1) 预处理
    data = model.data_preprocessor(dict(inputs=img), False)
    x = model.extract_feat(data['inputs'])
    # 2) 调用 RPN head
    proposal_list = model.rpn_head.predict(x, data_samples=None)[0]  # Tensor[N,4]
    proposals = proposal_list[:topk].cpu().numpy()
    # 3) 在原图上画框
    img_show = img.copy()
    for bbox in proposals:
        x1,y1,x2,y2 = map(int, bbox)
        img_show = mmcv.cv_imshow.imshow_bbox(img_show,
                            bboxes=[(x1,y1,x2,y2)],
                            colors='green', thickness=1)
    mmcv.imwrite(img_show, out_path)

def show_final_results(model, img_path, out_path, score_thr=0.5):
    """调用 inference_detector, 用 DetLocalVisualizer 可视化 bbox+mask。"""
    result = inference_detector(model, img_path)
    visualizer = DetLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend', out_dir=os.path.dirname(out_path))])
    # 构造 DataSample，便于调用
    data_sample = DetDataSample().set_metainfo(model.metainfo)
    data_sample.pred_instances = result
    visualizer.add_datasample(
        name='result',
        image=mmcv.imread(img_path),
        data_sample=data_sample,
        draw_pred=True,
        show=False,
        pred_score_thr=score_thr)
    # 输出文件名自动是 out_dir/name_instance.png

if __name__ == '__main__':
    # 配置区
    cfg_mask = Config.fromfile('configs/my_experiments/mask_rcnn_r50_voc_coco.py')
    ckpt_mask = 'work_dirs/mask_rcnn_r50_voc_coco/epoch_12.pth'
    model_mask = init_detector(cfg_mask, ckpt_mask, device='cuda:0')

    cfg_sparse = Config.fromfile('configs/my_experiments/sparse_rcnn_r50_voc_coco.py')
    ckpt_sparse = 'work_dirs/sparse_rcnn_r50_voc_coco/epoch_24.pth'
    model_sparse = init_detector(cfg_sparse, ckpt_sparse, device='cuda:0')

    # 选择四张测试图
    test_images = [
        'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000025.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000100.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000150.jpg',
    ]
    os.makedirs('work_dirs/vis/rpn_proposals', exist_ok=True)
    os.makedirs('work_dirs/vis/mrcnn_results', exist_ok=True)
    os.makedirs('work_dirs/vis/sparse_results', exist_ok=True)

    for img_path in test_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # 1) Mask R-CNN RPN proposals
        show_rpn_proposals(model_mask, img_path,
                           f'work_dirs/vis/rpn_proposals/{base}_mrcnn.png')
        # 2) Mask R-CNN final
        show_final_results(model_mask, img_path,
                           f'work_dirs/vis/mrcnn_results/{base}.png')
        # 3) Sparse R-CNN final
        show_final_results(model_sparse, img_path,
                           f'work_dirs/vis/sparse_results/{base}.png')
