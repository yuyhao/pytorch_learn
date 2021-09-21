# -*- coding: utf-8 -*-
# @Time: 2021/9/11 21:42
# @Author: yuyinghao
# @FileName: lstm.py
# @Software: PyCharm

import torch.nn as nn
import torch


class Config:
    seq_len = 30
    batch_size = 50
    input_size = 28
    hidden_size = 64
    num_layer = 2
    is_bi = True
    output_size = 10


class LSTM(nn.Module):
    """
    基础版LSTM
    """

    def __init__(self, input_size, hidden_size, num_layer, is_bi, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.is_bi = is_bi
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layer,
            bidirectional=self.is_bi
        )

        if self.is_bi:
            self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        output = self.lstm(input)
        output = self.fc(output[0][:, -1, :])

        return output


if __name__ == '__main__':
    input = torch.randn(Config.seq_len, Config.batch_size, Config.input_size)

    lstm = LSTM(
        Config.input_size,
        Config.hidden_size,
        Config.num_layer,
        Config.is_bi,
        Config.output_size)

    output = lstm(input)
