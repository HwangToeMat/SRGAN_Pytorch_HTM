import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.ResidualBlock1 = ResidualBlock(64)
        self.ResidualBlock2 = ResidualBlock(64)
        self.ResidualBlock3 = ResidualBlock(64)
        self.ResidualBlock4 = ResidualBlock(64)
        self.ResidualBlock5 = ResidualBlock(64)
        self.output_residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(64, 64 * 2 ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.pixel_shuffle2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2 ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.output =  nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        input = self.input(x)
        ResidualBlock1 = self.ResidualBlock1(input)
        ResidualBlock2 = self.ResidualBlock2(ResidualBlock1)
        ResidualBlock3 = self.ResidualBlock3(ResidualBlock2)
        ResidualBlock4 = self.ResidualBlock4(ResidualBlock3)
        ResidualBlock5 = self.ResidualBlock5(ResidualBlock4)
        output_residual = self.output_residual(ResidualBlock5)
        pixel_shuffle = self.pixel_shuffle(output_residual + input)
        pixel_shuffle2 = self.pixel_shuffle2(pixel_shuffle)
        output = self.output(pixel_shuffle2)
        output = (output+1)/2
        return output

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.Net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Net(x)
        return x.squeeze()
