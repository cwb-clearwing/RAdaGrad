#%%
from tqdm import tqdm
import torch
from torch import float16
import pandas as pd
from datetime import datetime

def full_count_params(NN,count_bias = False,with_grads = False):

    """ 
    Function that counts the total number of parameters needed for a full rank version of NN
    INPUTS:
    NN: neural network
    count_bias : flag variable, True if the biases are to be included in the total or not

    OUTPUTS:
    total_params : total number of parameters in the full rank version of NN
    """

    total_params = 0

    for l in NN.layer:

        n = str(l)

        if 'Linear' in n:

            total_params += 2*l.in_features*l.out_features if with_grads else l.in_features*l.out_features

            if count_bias and l.bias is not None:

                total_params += 2*len(l.bias) if with_grads else len(l.bias)

        if 'Conv' in n:

            total_params += 2*l.kernel_size_number*l.in_channels*l.out_channels if with_grads else l.kernel_size_number*l.in_channels*l.out_channels

            if count_bias and l.bias is not None:

                total_params += 2*len(l.bias) if with_grads else len(l.bias)

    return total_params


def count_params(T,with_grads = False):

    """ 
    function to count number of parameters inside a tensor
    INPUT:
    T : torch.tensor or None
    output:
    number of parameters contained in T
    """

    if len(T.shape)>1:

        if with_grads:

            return 2*int(torch.prod(torch.tensor(T.shape)))

        else:

            return int(torch.prod(torch.tensor(T.shape)))

    elif T == None:

        return 0

    else:

        if with_grads:

            return 2*T.shape[0]
        
        else:

            return T.shape[0]


def count_params_train(NN,count_bias = False,with_grads = False):

    """ 
    function to count the parameters in the train phase
    
    INPUTS:
    NN : neural network
    count_bias : flag variable, True if the biases are to be included in the total or not
    """

    total_params = 0

    for l in NN.layer:

        if hasattr(l,'lr') and l.lr:

            if not l.fixed:

                total_params += count_params(l.K[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.L[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.U[:,:l.dynamic_rank])
                total_params += count_params(l.V[:,:l.dynamic_rank])
                total_params += count_params(l.U_hat[:,:2*l.dynamic_rank])
                total_params += count_params(l.V_hat[:,:2*l.dynamic_rank])
                total_params += count_params(l.S_hat[:2*l.dynamic_rank,:2*l.dynamic_rank],with_grads)
                total_params += count_params(l.M_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                total_params += count_params(l.N_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                if count_bias:
                    total_params +=count_params(l.bias)

            else:

                total_params += count_params(l.K[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.L[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.U[:,:l.dynamic_rank])
                total_params += count_params(l.V[:,:l.dynamic_rank])
                total_params += count_params(l.S_hat[:2*l.dynamic_rank,:2*l.dynamic_rank],with_grads)
                total_params += count_params(l.M_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                total_params += count_params(l.N_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                if count_bias:
                    total_params +=count_params(l.bias)

        else:

            for n,p in l.named_parameters():

                if 'bias' not in n:

                    total_params += count_params(p,with_grads)   # add with gradsw

                elif 'bias' in n and count_bias:

                    total_params += count_params(p)

    return total_params


def count_params_test(NN,count_bias = False):

    """ 
    function to count the parameters in the test phase
    
    INPUTS:
    NN : neural network
    count_bias : flag variable, True if the biases are to be included in the total or not
    """

    total_params = 0

    for l in NN.layer:

        if hasattr(l,'lr') and l.lr:

            total_params += count_params(l.K[:,:l.dynamic_rank])
            total_params += count_params(l.L[:,:l.dynamic_rank])
            if count_bias:
                total_params +=count_params(l.bias)

        else:

            for n,p in l.named_parameters():

                if 'bias' not in n:

                    total_params += count_params(p)

                elif 'bias' in n and count_bias:

                    total_params +=count_params(p)

    return total_params

def accuracy(outputs,labels):

    return torch.mean(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = float16))