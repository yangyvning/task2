import re
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


def parse_logs(log_paths, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(output_dir)  # 只创建一个 Writer

    # 合并所有 epoch 的指标数据
    all_train_metrics = defaultdict(list)
    all_val_metrics = defaultdict(list)

    # 统一的正则表达式模式（保持原样）
    train_pattern = re.compile(
        r"Epoch\(train\)\s+\[(\d+)\]\[\s*\d+/\d+\]\s+lr: ([\d\.e+-]+).*?"
        r"loss: ([\d\.]+)\s+loss_rpn_cls: ([\d\.]+)\s+loss_rpn_bbox: ([\d\.]+)\s+"
        r"loss_cls: ([\d\.]+)\s+acc: ([\d\.]+)\s+loss_bbox: ([\d\.]+)\s+loss_mask: ([\d\.]+)"
    )

    val_pattern = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\]\s+"
        r"coco/bbox_mAP: ([\d\.]+)\s+coco/bbox_mAP_50: ([\d\.]+)\s+coco/bbox_mAP_75: ([\d\.]+)\s+"
        r"coco/segm_mAP: ([\d\.]+)\s+coco/segm_mAP_50: ([\d\.]+)\s+coco/segm_mAP_75: ([\d\.]+)"
    )

    # 处理所有日志文件
    for log_path in log_paths:
        print(f"Processing log file: {log_path}")
        with open(log_path, 'r', encoding='gbk', errors='ignore') as f:
            for line in f:
                line = line.strip()

                # 解析训练指标
                train_match = train_pattern.search(line)
                if train_match:
                    epoch = int(train_match.group(1))
                    all_train_metrics[epoch].append({
                        'loss': float(train_match.group(3)),
                        'loss_rpn_cls': float(train_match.group(4)),
                        'loss_rpn_bbox': float(train_match.group(5)),
                        'loss_cls': float(train_match.group(6)),
                        'acc': float(train_match.group(7)),
                        'loss_bbox': float(train_match.group(8)),
                        'loss_mask': float(train_match.group(9))
                    })

                # 解析验证指标
                val_match = val_pattern.search(line)
                if val_match:
                    epoch = int(val_match.group(1))
                    all_val_metrics[epoch].append({
                        'bbox_mAP': float(val_match.group(2)),
                        'bbox_mAP_50': float(val_match.group(3)),
                        'bbox_mAP_75': float(val_match.group(4)),
                        'segm_mAP': float(val_match.group(5)),
                        'segm_mAP_50': float(val_match.group(6)),
                        'segm_MAP_75': float(val_match.group(7))
                    })

    # 合并并写入 TensorBoard（按 epoch 排序）
    max_epoch = 0
    for epoch in sorted(all_train_metrics.keys()):
        # 取每个 epoch 最后一次记录的数据
        last_train = all_train_metrics[epoch][-1]
        writer.add_scalar('Train/loss_total', last_train['loss'], epoch)
        writer.add_scalar('Train/loss_rpn_cls', last_train['loss_rpn_cls'], epoch)
        writer.add_scalar('Train/loss_rpn_bbox', last_train['loss_rpn_bbox'], epoch)
        writer.add_scalar('Train/acc', last_train['acc'], epoch)
        max_epoch = max(max_epoch, epoch)

    for epoch in sorted(all_val_metrics.keys()):
        # 取每个 epoch 第一次验证结果（通常验证只进行一次）
        val_data = all_val_metrics[epoch][0]
        writer.add_scalar('Val/bbox_mAP', val_data['bbox_mAP'], epoch)
        writer.add_scalar('Val/segm_mAP', val_data['segm_mAP'], epoch)
        writer.add_scalar('Val/bbox_mAP_50', val_data['bbox_mAP_50'], epoch)
        writer.add_scalar('Val/segm_mAP_50', val_data['segm_mAP_50'], epoch)
        max_epoch = max(max_epoch, epoch)

    writer.close()
    print(f"Successfully merged logs up to epoch {max_epoch} in {output_dir}")


if __name__ == "__main__":
    # 需要合并的两个日志文件路径
    log_files = [
        r"C:\Users\HP\Desktop\mmdetection-main\work_dirs\sparse_rcnn_r50_voc_coco\20250520_185518\20250520_185518.log",
        r"C:\Users\HP\Desktop\mmdetection-main\work_dirs\sparse_rcnn_r50_voc_coco\20250521_140340\20250521_140340.log"
    ]

    parse_logs(
        log_paths=log_files,
        output_dir=os.path.join(os.getcwd(), "merged_tensorboard_logs")
    )