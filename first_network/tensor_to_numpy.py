# -*- coding: utf-8 -*-
# @Time: 2021/8/29 20:59
# @Author: yuyinghao
# @FileName: tensor_to_numpy.py
# @Software: PyCharm

import torch
import numpy as np

# 0 tensor和numpy之间的转换(使用cpu情况下，共享一个物理位置)
a = torch.ones(5)

b = a.numpy()
print(b)

a.add_(1)  # b也同时发生改变

# 1 numpy转tensor
a = np.ones((1, 2))
b = torch.from_numpy(a)

np.add(a, 1, out=a)
print(b)






