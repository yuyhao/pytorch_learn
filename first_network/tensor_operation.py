# -*- coding: utf-8 -*-
# @Time: 2021/8/29 20:38
# @Author: yuyinghao
# @FileName: tensor_operation.py
# @Software: PyCharm


import torch

# 0 生成张量
x = torch.rand(5, 3)
y = torch.rand(5, 3)

# 1 张量的加法
print(x + y)  # 各个位置的元素进行相加

print(torch.add(x, y))

result = torch.empty(5, 3)  # 加参数
torch.add(x, y, out=result)

# 2 加法的第二种方式(改变自身)
y.add_(x)
print(y)

# 3 取片
x[:, 1]

# 4 调整张量的形状
x = torch.randn(4, 4)
y = x.view(16)  # 将张量压缩成1维

z = x.view(-1, 8)  # 将张量根据列压缩, 行自动适应
print(z)

# 张量的大小
x = torch.randn(1)
print(x)
print(x.item())  # 值的大小
