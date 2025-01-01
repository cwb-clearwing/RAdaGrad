# imports 
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
import warnings
warnings.filterwarnings("ignore", category=Warning)

# low rank convolution class 

class Conv2d_lr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, rank = None, 
                 r_min = None, device = None)->None:

        """  
        Initializer for the convolutional low rank layer (filterwise), extention of the classical Pytorch's convolutional layer.
        INPUTS:
        in_channels: number of input channels (Pytorch's standard)
        out_channels: number of output channels (Pytorch's standard)
        kernel_size : kernel_size for the convolutional filter (Pytorch's standard)
        dilation : dilation of the convolution (Pytorch's standard)
        padding : padding of the convolution (Pytorch's standard)
        stride : stride of the filter (Pytorch's standard)
        bias  : flag variable for the bias to be included (Pytorch's standard)
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        dtype : Type of the tensors (Pytorch standard, to finish)
        """
        super(Conv2d_lr, self).__init__()
        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size,int) else kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = dilation if type(dilation)==tuple else (dilation, dilation)
        self.padding = padding if type(padding) == tuple else(padding, padding)
        self.stride = (stride if type(stride)==tuple else (stride, stride))
        self.in_channels = in_channels
        self.layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True, device=device)
        # Initialization
        self.rank = rank
        self.device = device
        self.init = True
        #self.weight = torch.nn.Parameter(torch.empty(tuple([self.out_channels, self.in_channels] +self.kernel_size),**factory_kwargs), requires_grad = True)
        #self.lr = True if self.rank!=None else False
        self.rmax = int(min([self.out_channels, self.in_channels*self.kernel_size_number]) / 2)
        self.layer.weight.r = min([rank,self.rmax])
        self.layer.weight.minimum_rank = r_min
        # Weight Parameters
        n,m = self.out_channels,self.in_channels*self.kernel_size_number
        self.rmax = min(math.floor(m/2)-1, math.floor(n/2)-1, self.rmax)
        U = torch.randn(n,self.rmax)
        V = torch.randn(m,self.rmax)
        U,_,_ = torch.linalg.svd(U)
        V,_,_ = torch.linalg.svd(V)
        Vh = V.transpose(0,1)
        _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(2*self.rmax))))
        self.layer.weight.s = torch.nn.Parameter(s_ordered.to(device) ,requires_grad=False)   
        self.layer.weight.u = torch.nn.Parameter(U[:,0:self.rmax*2].to(device) ,requires_grad=False) 
        self.layer.weight.vh = torch.nn.Parameter(Vh[0:self.rmax*2,:].to(device) ,requires_grad=False) 
        self.layer.weight.init = True
        self.layer.weight.data = self.layer.weight.u.matmul(torch.diag(self.layer.weight.s[0:self.rmax*2]).matmul(self.layer.weight.vh)).view(self.out_channels, self.in_channels, kernel_size, kernel_size)
        self.layer.bias.is_matrix = False
        self.device = device
        self.lr = True
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.layer.weight, a=math.sqrt(5))
         # for testing
        # self.original_weight = Parameter(self.weight.reshape(self.original_shape))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.layer.weight)
        #     if fan_in != 0:
        #         bound = 1 / math.sqrt(fan_in)
        #         init.uniform_(self.bias, -bound, bound)  
    
    def forward(self, x):
        y = self.layer(x)
        return y
