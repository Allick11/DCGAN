import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# 防止画图崩溃
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_data = torchvision.datasets.MNIST('data',
                                        train=True,
                                        transform=transforms,
                                        download=True)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)


# 生成器(100 -> 28*28)
# 用反卷积层将噪声放大
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Linear1 = nn.Linear(100, 256 * 7 * 7)  # 为什么是256*7*7，因为要生成28*28的图，7是28的倍数。
        self.bn1 = nn.BatchNorm1d(256 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)  # 这层还没进行缩放,是128*7*7
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1)  # (64,14,14)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=2, padding=1)  # (1*28*28)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = self.bn1(x)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.deconv1(x))
        x = self.bn2(x)
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = F.tanh(self.deconv3(x))
        return x


# 判别器(1 -> 128*7*7 ->1)
# 用卷积层
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 7 * 7, 1)  # 128*7*7经过测试输出得到

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.bn1(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.sigmoid(self.fc1(x))
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator().to(device)
dis = Discriminator().to(device)

# loss
loss = torch.nn.BCELoss()
g_optimizer = optim.Adam(gen.parameters(), lr=1e-5)
d_optimizer = optim.Adam(dis.parameters(), lr=1e-5)


# 绘图函数
def gen_img_plot(model, test_input):
    pre = np.squeeze(model(test_input).detach().cpu().numpy())  # 将预测值转成np数组形式
    fig = plt.figure(figsize=(4, 4))
    for i in range(pre.shape[0]):
        plt.subplot(4, 4, i + 1)
        # 之前Tanh输出的是(-1,1)之间的数，imshow需要（0,1）之间所以要把pre变成（0,1）之间，做法就是(pre[i]+1)/2
        plt.imshow((pre[i] + 1) / 2, cmap='gray')
        plt.axis('off')  # 不显示坐标轴
    plt.show()  # 这里一次会输出16张图片


# loss曲线函数
def plot_loss(D_loss, G_loss):
    # 将列表转换为 Tensor 对象
    D_loss_tensor = torch.tensor(D_loss)
    G_loss_tensor = torch.tensor(G_loss)
    # 数据还在 GPU 上，画图要搬到 CPU 上来
    plt.plot(D_loss_tensor.cpu().detach().numpy(), label='Discriminator Loss')
    plt.plot(G_loss_tensor.cpu().detach().numpy(), label='Generator Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# 训练部分
# 16个100维的噪声
epochs = 100
test_input = torch.randn(16, 100, device=device)
D_loss = []
G_loss = []
for epoch in range(epochs):
    D_epoch_loss = 0
    G_epoch_loss = 0
    count = len(train_dataloader)  # len(dataloader)返回批次数   len(dataset)返回样本数
    for step, (img, _) in enumerate(train_dataloader):
        img = img.to(device)
        size = img.size(0)  # 得到每批的图片个数,根据图片个数来确定输入噪声个数
        noise = torch.randn(size, 100, device=device)
        # D对真图的loss
        d_optimizer.zero_grad()
        real_output = dis(img)
        d_real_loss = loss(real_output, torch.ones_like(real_output))
        d_real_loss.backward()
        # D对假图的loss
        gen_img = gen(noise)
        fake_output = dis(gen_img.detach())  # 这里需要优化判别器，所以用gen_img.detach()对生成器的梯度做截断
        d_fake_loss = loss(fake_output, torch.zeros_like(fake_output))
        d_fake_loss.backward()
        # D的loss
        d_loss = d_fake_loss + d_real_loss
        d_optimizer.step()

        # G的loss
        g_optimizer.zero_grad()
        fake_output = dis(gen_img)
        g_loss = loss(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        g_optimizer.step()

        # 求每一步累加的loss(不需要梯度)
        with torch.no_grad():
            D_epoch_loss += d_loss.item()  # .item()可转化为python的float类型
            G_epoch_loss += g_loss.item()
    # 求平均loss
    with torch.no_grad():
        D_epoch_loss /= count
        G_epoch_loss /= count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        print('Epoch: ', epoch + 1, 'D_loss: ', D_epoch_loss, 'G_loss: ', G_epoch_loss)
# 绘图
plot_loss(D_loss, G_loss)
gen_img_plot(gen, test_input)

# 保存生成器模型(只保存了参数)
torch.save(gen.state_dict(), 'generator.pth')
# 保存判别器模型
torch.save(dis.state_dict(), 'discriminator.pth')
