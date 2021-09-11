# -*- coding: utf-8 -*-
# @Time: 2021/8/29 20:27
# @Author: yuyinghao
# @FileName:
# @Software: PyCharm

# form __future__ import print_function
import torch

# 0 生成张量(不全为0)
x = torch.empty(5, 3)  # 生成5*3的矩阵
print(x)

# 1 生成均匀分布的初始化张量
x = torch.rand(5, 3)  # [0, 1]
print(x)

# 2 生成全0张量
x = torch.zeros(5, 3)
print(x)

# 3 从已有矩阵转化为张量
x = torch.tensor([[1, 2], [2, 4]])
print(x)

# 4 从已有张量中创建一个张量
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

# 获取张量的形状
x.size()