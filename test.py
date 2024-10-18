import torch
print(torch.cuda.is_available())  # 检查是否检测到 GPU
print(torch.cuda.device_count())  # 查看可用 GPU 数量
print(torch.cuda.current_device())  # 查看当前使用的 GPU
print(torch.cuda.get_device_name(0))  # 查看 GPU 名称

