#%%
# import custom layers
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS_DLRT.my_conv import Conv2d_lr
from optimizer_KLS_DLRT.Linear_layer_lr_new import Linear
import torch


class VGG16(torch.nn.Module):
    def __init__(self, device='cpu'):
        """
        VGG16
        """
        super(VGG16, self).__init__()
        self.device = device

        self.layer = torch.nn.Sequential(
            # Block 1
            Conv2d_lr(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, rank=64, device=self.device),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, rank=64, device=self.device),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            Conv2d_lr(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, rank=128, device=self.device),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, rank=128, device=self.device),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            Conv2d_lr(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, rank=256, device=self.device),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, rank=256, device=self.device),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, rank=256, device=self.device),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            Conv2d_lr(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, device=self.device),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, device=self.device),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, device=self.device),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, device=self.device),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, device=self.device),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            Conv2d_lr(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, rank=512, device=self.device),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.AvgPool2d(kernel_size=1, stride=1),

            # Flatten
            torch.nn.Flatten(),

            # Fully Connected Layers
            Linear(512 * 1 * 1, out_features=4096, rank=4096, device=self.device),
            torch.nn.ReLU(),
            Linear(4096, out_features=4096, rank=4096, device=self.device),
            torch.nn.ReLU(),
            Linear(4096, out_features=10, rank=100, device=self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def update_step(self, new_step='K'):
        for l in self.layer:
            if hasattr(l, 'lr') and l.lr:
                l.step = new_step

    def populate_gradients(self, x, y, criterion, step='all'):
        if step == 'all':
            self.update_step(new_step='K')
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            self.update_step(new_step='L')
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            return loss, output.detach()
        else:
            self.update_step(new_step=step)
            loss = criterion(self.forward(x), y)
            return loss


# import numpy as np
# NN = VGG16()
# print([(n, p.requires_grad) for n, p in NN.named_parameters()])
# x = torch.randn((1, 3, 32, 32))
# y = torch.tensor(np.random.choice(range(10), 1))
# NN.populate_gradients(x, y, torch.nn.CrossEntropyLoss())
# print([(n, p.grad is not None) for n, p in NN.named_parameters()])
# %%