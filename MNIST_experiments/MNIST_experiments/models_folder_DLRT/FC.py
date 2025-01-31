#%%
# import custom layers
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS_DLRT.Linear_layer_lr_new import Linear
import torch

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
            Linear(28*28,out_features = 5120, rank = 500,device = self.device), 
            # 500->400->300->200->100 test train results/accuracy time charts more epochs 
            # charts : baseline, DLRT, RGD
            # Dynamic rank 
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            Linear(5120,out_features = 5120, rank = 500,device = self.device), 
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            Linear(5120,out_features = 5120, rank = 500,device = self.device),
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            Linear(5120,out_features = 5120, rank = 500,device = self.device),
            torch.nn.BatchNorm1d(5120),
            torch.nn.ReLU(),
            # torch.nn.ReLU(),
            # torch.nn.Flatten(), # 800
            Linear(5120,out_features = 10,rank = 10, device = self.device)
        )
        #for layer in self.layer:
        #    if hasattr(layer,'weight') and hasattr(layer,'bias'):
        #        layer.weight.is_matrix = True
        #        layer.bias.is_matrix = False

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def update_step(self,new_step = 'K'):
        for l in self.layer:
            if hasattr(l,'lr') and l.lr:
                l.step = new_step

    def populate_gradients(self,x,y,criterion,step = 'all'):

        if step == 'all':
        
            self.update_step(new_step = 'K')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            self.update_step(new_step = 'L')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            return loss,output.detach()

        else:
            
            self.update_step(new_step = step)
            loss = criterion(self.forward(x),y)
            return loss

# import numpy as np
# NN = Lenet5()
# print([(n,p.requires_grad) for n,p in NN.named_parameters()])
# x= torch.randn((1,1,28,28))
# y = torch.tensor(np.random.choice(range(10),1))
# # NN.populate_gradients(x,y,torch.nn.CrossEntropyLoss())
# # print([(n,p.grad is not None) for n,p in NN.named_parameters()])
# %%
