# -*- coding: utf-8 -*-
# @Time: 2021/9/6 23:28
# @Author: yuyinghao
# @FileName: fc_net_class.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

"""
构建网络
"""


class FullConnectClassifier(nn.Module):
    def __init__(self):
        super(FullConnectClassifier, self).__init__()
        self.first_layer = nn.Linear(100, 60)
        self.activate_func = nn.ReLU()
        self.second_layer = nn.Linear(60, 40)
        self.activate_func_2 = nn.ReLU()
        self.third_layer = nn.Linear(40, 10)
        self.activate_func_3 = nn.Softmax(1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.activate_func(x)
        x = self.second_layer(x)
        x = self.activate_func_2(x)
        x = self.third_layer(x)
        x = self.activate_func_3(x)

        return x


if __name__ == '__main__':
    # 构造数据
    x = torch.randn(2000, 100)
    y = torch.randn(2000, 10)

    # 初始化网络
    fcc = FullConnectClassifier()

    # 参数
    epoch = 20
    batch_size = 50

    # 优化器
    optimizer = optim.Adam(fcc.parameters(), lr=0.001)

    # 损失函数
    loss_func = nn.CrossEntropyLoss()

    # 开始训练
    running_loss = 0
    for i in range(epoch):
        # 重要的一步，打乱顺序
        index = torch.randperm(x.size(0))
        x = x[index]
        y = x[index]
        batch = range(0, x.size(0), batch_size)

        for idx, v in enumerate(batch):
            inputs = x[v: v + batch_size, :]
            target = y[v: v + batch_size, :]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            optimizer.zero_grad()  # 清空梯度

            outputs = fcc(inputs)

            loss = loss_func(outputs, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i != 0 and i % 2 == 0:
                print(
                    'epoch:{} | batch:{}| loss:{:.5f}'.format(
                        i, j, running_loss / 2))
                running_loss = 0.0
