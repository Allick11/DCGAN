import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
# 防止画图崩溃
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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


'''
#在CPU上加载
gen = Generator()
device = torch.device("cpu")
gen.load_state_dict(torch.load('generator.pth', map_location=device))
gen.to(device)
'''

# 在GPU上加载
gen = Generator()
device = "cuda" if torch.cuda.is_available() else "cpu"
gen.load_state_dict(torch.load('generator.pth', map_location=device))
gen.to(device)

gen.eval()
noise = torch.randn(16, 100, device=device)

# 使用生成器生成样本
with torch.no_grad():
    generated_samples = gen(noise).cpu()
# 可视化生成的图像
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    sample = generated_samples[i].permute(1, 2, 0)
    ax.imshow((sample + 1) / 2, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
