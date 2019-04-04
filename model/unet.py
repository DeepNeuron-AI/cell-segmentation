import numpy as np
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname[:4] == 'Conv':
        N = m.kernel_size[0] ** 2 * m.in_channels
        std = np.sqrt(2 / N)
        m.weight.data.normal_(mean=0.0, std=std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # down
        self.con1 = DoubleConv(n_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self._down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = DoubleConv(512, 1024)

        # up
        self.up6 = Up(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = Up(512, 256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = Up(256, 128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = Up(128, 64)
        self.conv9 = nn.Sequential(
            DoubleConv(128, 64),
            nn.Conv2d(64, n_classes, kernel_size=1, padding=0),
            nn.BatchNorm2d(n_classes),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, x):
        # down
        x0 = self.con1(x)
        x1 = self._down(x0)
        x1 = self.conv2(x1)
        x2 = self._down(x1)
        x2 = self.conv3(x2)
        x3 = self._down(x2)
        x3 = self.conv4(x3)
        xb = self._down(x3)
        xb = self.conv5(xb)

        # Up
        x6 = self.up6(xb)
        x7 = torch.cat([x3, x6], dim=1)
        x7 = self.conv6(x7)
        x7 = self.up7(x7)
        x8 = torch.cat([x2, x7], dim=1)
        x8 = self.conv7(x8)
        x8 = self.up8(x8)
        x9 = torch.cat([x1, x8], dim=1)
        x9 = self.conv8(x9)
        x9 = self.up9(x9)
        x10 = torch.cat([x0, x9], dim=1)
        x10 = self.conv9(x10)
        return x10


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # state size. (out_ch) x (in_ch - 2)^2

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            # state size. (out_ch) x (in_ch - 4)^2
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # state size. (out_ch) x (in_ch * 2)^2
        )

    def forward(self, x):
        x = self.conv_up(x)
        return x
