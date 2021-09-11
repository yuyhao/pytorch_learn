# -*- coding: utf-8 -*-
# @Time: 2021/9/9 0:00
# @Author: yuyinghao
# @FileName: rnn_net.py
# @Software: PyCharm

import torch.nn as nn


class Config:
    batch_size = 50


class RNN:
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=2,
            bidirectional=True
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h, c) = self.rnn(x)
        out = self.out(r_out[:, -1, :])

        return out
