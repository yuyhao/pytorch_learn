# -*- coding: utf-8 -*-
# @Time: 2021/8/29 21:13
# @Author: yuyinghao
# @FileName: auto_differential.py
# @Software: PyCharm

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)  # 标准正态分布
a = a * 3 / (a - 1)

print(a.requires_grad)
a.requires_grad_(True)

b = (a * a).sum()
print(b.grad_fn)


######################梯度##########################
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)