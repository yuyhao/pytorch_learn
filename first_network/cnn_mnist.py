# -*- coding: utf-8 -*-
# @Time: 2021/9/1 22:07
# @Author: yuyinghao
# @FileName: cnn_mnist.py
# @Software: PyCharm

"""
教程
https://blog.csdn.net/m0_60720377/article/details/119957268?utm_medium=distribute.pc_relevant_t0.580420
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision # 数据库模块
import matplotlib.pyplot as plt

# 训练轮数，训练次数越多，精度越高
EPOCH = 2
BATCH_SIZE = 50  # 每次训练的样本个数
LR = 0.01  # 学习率
DOWNLOAD_MNIST = False
DATA_FILE = r'E:\0_工具\5_学习\pytorch_study\first_network\data'

# 下载训练集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成tensor
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就会自动下载数据集,当等于true
)

# Mnist 手写数字 测试集
test_data = torchvision.datasets.MNIST(
	root='./mnist/',
	train=False, # this is training data
    download=DOWNLOAD_MNIST
)

# 定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 通道数
                out_channels=16,  # 输出通道数
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )  # 定义第一层卷积网络

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    # 前馈神经网络
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
# 加载数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                               shuffle=True)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]

# 训练神经网络
cnn = CNN()  # 创建cnn
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 优化算法
loss_func = nn.CrossEntropyLoss()  # 损失函数

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 每一步 loader 释放50个数据用来学习
        #b_x = b_x.cuda() # 若有cuda环境，取消注释
        #b_y = b_y.cuda() # 若有cuda环境，取消注释
        output = cnn(b_x)  # 输入一张图片进行神经网络训练
        loss = loss_func(output, b_y)  # 计算神经网络的预测值与实际的误差
        optimizer.zero_grad()  #将所有优化的torch.Tensors的梯度设置为零
        loss.backward()  # 反向传播的梯度计算
        optimizer.step()  # 执行单个优化步骤
        if step % 50 == 0: # 我们每50步来查看一下神经网络训练的结果
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # 若有cuda环境，使用84行，注释82行
            # pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = float((pred_y == test_y).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data,
            	'| test accuracy: %.2f' % accuracy)


# 训练完成后进行测试
# test 神经网络
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
# 若有cuda环境，使用92行，注释90行
#pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
# save CNN
# 仅保存CNN参数，速度较快
torch.save(cnn.state_dict(), './model/CNN_NO1.pk')
# 保存CNN整个结构
#torch.save(cnn(), './model/CNN.pkl')


