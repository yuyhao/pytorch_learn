# -*- coding: utf-8 -*-
# @Time: 2021/9/9 0:00
# @Author: yuyinghao
# @FileName: lstm_net.py
# @Software: PyCharm

"""
主题: pytorch lstm网络构造
来源1 lstm参数解析: https://zhuanlan.zhihu.com/p/79064602
来源2 lstm实例: https://blog.csdn.net/Leon_winter/article/details/92592622
"""

import torch.nn as nn
import torch


class Config:
    batch_size = 50
    seq_max = 10


class LSTMT(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=28,  # embedding的维度
            hidden_size=64,  # 一个LSTM有多少个神经元
            num_layers=3,  # LSTM的个数
            bidirectional=True,  # 是否是双向
            batch_first=False  # 第一个维度是否表示batch,如果为False的话则第一个维度表示seq_len
        )

        self.out = nn.Linear(128, 10)

    def forward(self, x):
        r_out, (h, c) = self.lstm(x)
        out = self.out(r_out[:, -1, :])

        return out


if __name__ == '__main__':
    # 0 输入
    input = torch.randn(Config.seq_max, Config.batch_size, 28)
    h0 = torch.randn(6, Config.batch_size, 64)
    c0 = torch.randn(6, Config.batch_size, 64)

    # 1 实例化构建的网络
    lstm = LSTM()
    output = lstm(input)

    # 2 直接用函数
    lstm = nn.LSTM(
        input_size=28,
        hidden_size=64,
        num_layers=3,
        bidirectional=True)
    output2 = lstm(input, (h0, c0))
