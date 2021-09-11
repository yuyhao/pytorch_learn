# -*- coding: utf-8 -*-
# @Time: 2021/9/11 20:07
# @Author: yuyinghao
# @FileName:
# @Software: PyCharm

import torch

# pack: 将batch的seq合并成一维
from torch.nn.utils.rnn import pack_sequence

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5])
c = torch.tensor([6])
pack_res = pack_sequence([a, b, c])  # res: data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1])

# pad
from torch.nn.utils.rnn import pad_sequence

a = torch.ones(25, 30)
b = torch.ones(22, 30)
c = torch.ones(15, 30)
pad_res = pad_sequence([a, b, c])
pad_res.size()
