import re
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


def parse_mask_rcnn_single_log(log_path, output_dir):
    """解析单个Mask R-CNN日志文件生成可视化数据"""
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)

    # 初始化数据结构
    metrics = defaultdict(lambda: {
        'train_loss': [],
        'val_bbox': defaultdict(list),
        'val_segm': defaultdict(list)
    })

    # ===== 正则表达式规则 =====
    train_pattern = re.compile(
        r"Epoch\(train\)\s+\[(\d+)\].*?loss:\s+([\d\.]+)"
    )

    val_pattern = re.compile(
        r"Epoch\(val\)\s+\[(\d+)\].*?"
        r"coco/bbox_mAP:\s+([\d\.]+).*?"
        r"coco/bbox_mAP_50:\s+([\d\.]+).*?"
        r"coco/bbox_mAP_75:\s+([\d\.]+).*?"
        r"coco/bbox_mAP_s:\s+([\d\.]+).*?"
        r"coco/bbox_mAP_m:\s+([\d\.]+).*?"
        r"coco/bbox_mAP_l:\s+([\d\.]+).*?"
        r"coco/segm_mAP:\s+([\d\.]+).*?"
        r"coco/segm_mAP_50:\s+([\d\.]+).*?"
        r"coco/segm_mAP_75:\s+([\d\.]+).*?"
        r"coco/segm_mAP_s:\s+([\d\.]+).*?"
        r"coco/segm_mAP_m:\s+([\d\.]+).*?"
        r"coco/segm_mAP_l:\s+([\d\.]+)"
    )

    # ===== 解析日志内容 =====
    print(f"🔍 正在解析日志文件: {log_path}")
    with open(log_path, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析训练损失
            if 'Epoch(train)' in line:
                train_match = train_pattern.search(line)
                if train_match:
                    epoch = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    metrics[epoch]['train_loss'].append(loss)

            # 解析验证指标
            elif 'Epoch(val)' in line:
                val_match = val_pattern.search(line)
                if val_match:
                    epoch = int(val_match.group(1))
                    # 提取bbox指标
                    metrics[epoch]['val_bbox']['mAP'].append(float(val_match.group(2)))
                    metrics[epoch]['val_bbox']['mAP_50'].append(float(val_match.group(3)))
                    metrics[epoch]['val_bbox']['mAP_75'].append(float(val_match.group(4)))
                    metrics[epoch]['val_bbox']['AP_s'].append(float(val_match.group(5)))
                    metrics[epoch]['val_bbox']['AP_m'].append(float(val_match.group(6)))
                    metrics[epoch]['val_bbox']['AP_l'].append(float(val_match.group(7)))

                    # 提取segm指标
                    metrics[epoch]['val_segm']['mAP'].append(float(val_match.group(8)))
                    metrics[epoch]['val_segm']['mAP_50'].append(float(val_match.group(9)))
                    metrics[epoch]['val_segm']['mAP_75'].append(float(val_match.group(10)))
                    metrics[epoch]['val_segm']['AP_s'].append(float(val_match.group(11)))
                    metrics[epoch]['val_segm']['AP_m'].append(float(val_match.group(12)))
                    metrics[epoch]['val_segm']['AP_l'].append(float(val_match.group(13)))

    # ===== 写入TensorBoard =====
    print("\n🔄 正在生成可视化数据...")
    for epoch in sorted(metrics.keys()):
        # 训练损失（取平均值）
        if metrics[epoch]['train_loss']:
            avg_loss = sum(metrics[epoch]['train_loss']) / len(metrics[epoch]['train_loss'])
            writer.add_scalar('0_Train/Loss', avg_loss, epoch)

        # 验证指标（取最后一次记录）
        if metrics[epoch]['val_bbox']:
            writer.add_scalar('1_Validation/bbox/mAP', metrics[epoch]['val_bbox']['mAP'][-1], epoch)
            writer.add_scalar('1_Validation/bbox/mAP_50', metrics[epoch]['val_bbox']['mAP_50'][-1], epoch)
            writer.add_scalar('1_Validation/bbox/mAP_75', metrics[epoch]['val_bbox']['mAP_75'][-1], epoch)
            writer.add_scalar('1_Validation/bbox/AP_small', metrics[epoch]['val_bbox']['AP_s'][-1], epoch)
            writer.add_scalar('1_Validation/bbox/AP_medium', metrics[epoch]['val_bbox']['AP_m'][-1], epoch)
            writer.add_scalar('1_Validation/bbox/AP_large', metrics[epoch]['val_bbox']['AP_l'][-1], epoch)

        if metrics[epoch]['val_segm']:
            writer.add_scalar('2_Validation/segm/mAP', metrics[epoch]['val_segm']['mAP'][-1], epoch)
            writer.add_scalar('2_Validation/segm/mAP_50', metrics[epoch]['val_segm']['mAP_50'][-1], epoch)
            writer.add_scalar('2_Validation/segm/mAP_75', metrics[epoch]['val_segm']['mAP_75'][-1], epoch)
            writer.add_scalar('2_Validation/segm/AP_small', metrics[epoch]['val_segm']['AP_s'][-1], epoch)
            writer.add_scalar('2_Validation/segm/AP_medium', metrics[epoch]['val_segm']['AP_m'][-1], epoch)
            writer.add_scalar('2_Validation/segm/AP_large', metrics[epoch]['val_segm']['AP_l'][-1], epoch)

    writer.close()
    print(f"✅ 数据已保存至：{os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # ===== 配置参数 =====
    log_file = r"work_dirs\mask_rcnn_r50_voc_coco\20250519_152917\20250519_152917.log"
    output_dir = os.path.join(os.getcwd(), "mask_rcnn_tensorboard_single")

    # ===== 执行解析 =====
    parse_mask_rcnn_single_log(
        log_path=log_file,
        output_dir=output_dir
    )