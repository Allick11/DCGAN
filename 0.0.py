import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader



'''
# 如果不懂卷积（反卷积）过后的图维度是多少可直接调用这层假设一个图来输出
layer = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=1)
input = torch.randn(1, 1, 28, 28)  # 假设一张（28*28的图作为输入,这里第一个维度是batch_size，第二个维度是通道数）
out = layer(input)
print(out.shape)
'''

# 如果不懂某一层卷积（反卷积）过后的图维度是多少可直接调用这层假设一个图来输出
layer1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=1)
layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
bn1 = nn.BatchNorm2d(128)
input = torch.randn(1, 1, 28, 28)  # 假设一张（28*28的图作为输入,这里第一个维度是batch_size，第二个维度是通道数）
out1 = layer1(input)
out2 = layer2(out1)
out3 = bn1(out2)
print(out3.shape)  #torch.Size([1, 128, 7, 7])
