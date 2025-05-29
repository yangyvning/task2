import os
from torch.utils.tensorboard import SummaryWriter


def log_segm_mAP50_to_tensorboard(log_file_path, output_dir='./tensorboard_logs'):
    """
    Parse training logs and write segm_mAP_50 metrics to TensorBoard.

    Args:
        log_file_path (str): Path to the training log file
        output_dir (str): Directory to save TensorBoard logs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir)

    with open(log_file_path, 'r') as f:
        for line in f:
            if 'Epoch(val)' in line and 'segm_mAP_50:' in line:
                # Extract epoch number
                epoch_part = line.split('Epoch(val) [')[1].split(']')[0]
                epoch = int(epoch_part.split()[0])  # Get the first part before any /

                # Extract segm_mAP_50 value
                segm_mAP_50 = float(line.split('segm_mAP_50:')[1].split()[0])

                # Write to TensorBoard
                writer.add_scalar('val/segm_mAP_50', segm_mAP_50, epoch)

    writer.close()
    print(f"TensorBoard logs saved to {output_dir}")


# Your specific log file path
log_file_path = r"C:\Users\HP\Desktop\mmdetection-main\work_dirs\mask_rcnn_r50_voc_coco\20250519_152917\20250519_152917.log"

# Call the function with your log file
log_segm_mAP50_to_tensorboard(log_file_path)

print("Processing complete. You can now launch TensorBoard to view the results.")