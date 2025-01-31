import sys,os
import torch
from shampoo import LowRank_Linear
from new_conv import Conv2d_lr
sys.path.insert(1, os.path.join(sys.path[0], '..'))

class FC(torch.nn.Module):
    def __init__(self,device = 'cpu'):
        """  
        initializer for Full Connection Network.
        NEEDED ATTRIBUTES TO USE dlr_opt:
        self.layer
        NEEDED METHODS TO USE dlr_opt:
        self.forward : standard forward of the NN
        self.update_step : updates the step of all the low rank layers inside the neural net
        self.populate_gradients : method used to populate the gradients inside the neural network in one unique function
        """
        super(FC, self).__init__()
        self.device = device
        self.layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            LowRank_Linear(28*28,out_features = 5120, rank = 500, r_min = 5, device = self.device), 
            # 500->400->300->200->100 test train results/accuracy time charts more epochs 
            # charts : baseline, DLRT, RGD
            # Dynamic rank 
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            LowRank_Linear(5120,out_features = 5120, rank = 500, r_min = 5, device = self.device), 
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            LowRank_Linear(5120,out_features = 5120, rank = 500, r_min = 5,device = self.device),
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            LowRank_Linear(5120,out_features = 5120, rank = 500, r_min = 5,device = self.device),
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            # torch.nn.ReLU(),
            # torch.nn.Flatten(), # 800
            LowRank_Linear(5120,out_features = 10,rank = 10, r_min = 10,device = self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def populate_gradients(self,x,y,criterion):
        output = self.forward(x)
        loss = criterion(output,y)
        return loss,output.detach()
    
class Lenet5(torch.nn.Module):
    def __init__(self,device = 'cpu'):
        """  
        initializer for Full Connection Network.
        NEEDED ATTRIBUTES TO USE dlr_opt:
        self.layer
        NEEDED METHODS TO USE dlr_opt:
        self.forward : standard forward of the NN
        self.update_step : updates the step of all the low rank layers inside the neural net
        self.populate_gradients : method used to populate the gradients inside the neural network in one unique function
        """
        super(Lenet5, self).__init__()
        self.device = device
        self.layer = torch.nn.Sequential(
            Conv2d_lr(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1, rank = 20, r_min=10, device = self.device),  # 20 * 10 * 10 = 2000
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2),
            Conv2d_lr(in_channels = 20, out_channels = 50, kernel_size = 5, stride = 1, rank = 50, r_min=10, device = self.device),  # 50 * 8 * 8 = 3200
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride=2), 
            torch.nn.Flatten(), # 800
            LowRank_Linear(800,out_features = 500,rank = 500, r_min=10, device = self.device),  # 500
            torch.nn.ReLU(),
            LowRank_Linear(500,out_features = 10, rank = 10, r_min=10, device = self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def populate_gradients(self,x,y,criterion):
        output = self.forward(x)
        loss = criterion(output,y)
        return loss,output.detach()