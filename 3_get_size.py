import os

def get_file_size_mb(file_path):
    """获取文件大小并转换为MB"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)

print(get_file_size_mb("b_retinanet-pytorch-master/logs/best_epoch_weights.pth"))