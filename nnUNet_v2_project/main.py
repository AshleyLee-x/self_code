import torch

# 查看cuda是否可用
print(torch.cuda.is_available())
# 返回当前设备索引
print(torch.cuda.current_device())
# 返回GPU的数量
print(torch.cuda.device_count())
# 返回gpu的名字，设备索引默认从0开始
print(torch.cuda.get_device_name(0))