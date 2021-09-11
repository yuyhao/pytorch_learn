# -*- coding: utf-8 -*-
# @Time: 2021/9/5 23:26
# @Author: yuyinghao
# @FileName: fc_net.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

class FullConnect(nn.Module):
    """
    搭建一个简单的全连接神经网络
    """
    def __init__(self):
        super(FullConnect, self).__init__()
        self.layer_first = nn.Linear(10, 3)  # 第一层
        self.activate_first = nn.ReLU()  # 激活函数
        self.layer_second = nn.Linear(3, 1)  # 第二层

    def forward(self, x):
        x = self.layer_first(x)
        x = self.activate_first(x)
        x = self.layer_second(x)

        return x

class FullConnect2(nn.Module):
    def __int__(self):
        super(FullConnect2, self).__int__()
        self.seq = nn.Sequential(
            nn.Linear(10, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        x = self.seq(x)

        return x



if __name__ == '__main__':

    # 0 简单测试网络
    net = FullConnect()
    print(net)

    # 1 训练
    # 输入数据
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    # 配置损失函数
    loss_function = nn.MSELoss()  # 回归任务，选择MSE损失函数

    running_loss = 0
    for i in range(10):
        index = torch.randperm(100)  # 打乱顺序

        x = x[index]
        y = y[index]

        b = list(range(0, 100, 10))
        for j, b_index in enumerate(b):
            inputs = x[b_index: b_index + 10, :]
            target = y[b_index: b_index + 10, :]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = net(inputs)

            loss = loss_function(output, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i != 0 and i % 2 == 0:
                print('epoch:{} | batch:{}| loss:{:.5f}'.format(i, j, running_loss / 2))
                running_loss = 0.0

            torch.save(net.state_dict(),  './model/net.pk')





