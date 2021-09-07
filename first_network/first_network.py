# -*- coding: utf-8 -*-
# @Time: 2021/8/30 23:36
# @Author: yuyinghao
# @FileName: first_network.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    # 继承原有模型
    super(Net, self).__init__()
