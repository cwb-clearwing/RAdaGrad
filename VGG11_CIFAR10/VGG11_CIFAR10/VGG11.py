import sys,os
import torch
from shampoo import LowRank_Linear
from new_conv import Conv2d_lr
from torch import nn
sys.path.insert(1, os.path.join(sys.path[0], '..'))

class VGG11(torch.nn.Module):
    def __init__(self, device='cpu'):
        """
        VGG11
        """
        super(VGG11, self).__init__()
        self.device = device

        self.layer = torch.nn.Sequential(
            Conv2d_lr(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, rank=64, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d_lr(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, rank=128, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d_lr(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, rank=256, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, rank=256, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d_lr(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, r_min=10,
                      device=self.device),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),


            nn.Flatten(),
            LowRank_Linear(in_features=512, out_features=4096, rank=512, r_min=10, device=self.device),
            nn.ReLU(inplace=True),
            LowRank_Linear(in_features=4096, out_features=4096, rank=4096, r_min=10, device=self.device),
            nn.ReLU(inplace=True),
            LowRank_Linear(in_features=4096, out_features=10, rank=100, r_min=10, device=self.device)
        )

    def forward(self, x):

        x = self.layer(x)
        return x

    def populate_gradients(self, x, y, criterion):

        output = self.forward(x)  
        loss = criterion(output, y)
        return loss, output.detach()