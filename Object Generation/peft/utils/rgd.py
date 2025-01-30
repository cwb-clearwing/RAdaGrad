import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import numpy as np
import time
import scipy

class RGD_Opt(Optimizer):
    def __init__(self, params, lr=1e-1, momentum=0, weight_decay=0,epsilon=0.01,theta=0.1,pre=False):
        """

        :param params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param weight_decay: (float, optional): weigt decay factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional): update frequency to compute inverse (default: 1)
        """
        
        #defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,epsilon=epsilon,timer1=0,timer2=0,theta=theta,pre=pre,integrator=torch.optim.SGD(params,lr=lr))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,epsilon=epsilon,timer1=0,timer2=0,theta=theta,pre=pre)
        super(RGD_Opt, self).__init__(params, defaults)
        
    def integration_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p,'is_matrix') and p.is_matrix:
                    temp = p.grad.detach().clone()
                    p.data = p.data - group["lr"] * temp

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            pre = group['pre']
            loss = None
            if closure is not None:
                with torch.set_grad_enabled(True):
                    loss = closure()
                    loss.backward()
            if pre:
                self.preprocess_step(pre)
            self.integration_step()
            self.postprocess_step(pre)
        return loss

    def preprocess_step(self,pre):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                if hasattr(p,'is_matrix') and p.is_matrix:
                    temp = p.grad.detach().clone()
                    original_weight_shape = temp.shape
                    temp = temp.view(temp.size(0), -1)
                    n,m = temp.size()    
                    if pre:
                        state["pos_l"] = ((group["epsilon"]) * torch.eye(n=n, device=p.device) + torch.diag(torch.diag(temp.matmul(temp.t()), 0))) ** (+1/4)
                        state["neg_l"] = torch.diag(torch.diag(state["pos_l"], 0) ** (-1))
                        state["pos_r"] = ((group["epsilon"]) * torch.eye(n=m, device=p.device) + torch.diag(torch.diag(temp.t().matmul(temp), 0))) ** (+1/4)
                        state["neg_r"] = torch.diag(torch.diag(state["pos_r"], 0) ** (-1))
                        p.grad = (state["neg_l"].matmul(temp.matmul(state["neg_r"]))).view(original_weight_shape)
                    else:
                        state["pos_l"] = torch.eye(n=n, device=p.device)
                        state["neg_l"] = state["pos_l"]
                        state["neg_r"] = torch.eye(n=m, device=p.device)
                        state["pos_r"] = state["neg_r"]

    def postprocess_step(self,pre):
        for group in self.param_groups:
            for p in group["params"]:
                original_shape = p.grad.shape
                Dim2_shape = p.grad.view(p.grad.size(0), -1).size()
                n,m = Dim2_shape
                if not hasattr(p,'is_matrix') or not p.is_matrix:
                    continue
                grad = p.grad.clone()
                state = self.state[p]
                if p.s == None:
                    state["step"] = 0
                    if hasattr(p,'is_matrix') and p.is_matrix: #and hasattr(p,'init') and not p.init:
                        state["svd"] = torch.linalg.svd(p.data.view(Dim2_shape))
                        u, s ,vh = state["svd"]
                        p.s = s
                        p.u = u
                        p.vh = vh
                        # 近似
                        p.data = (u[:,0:p.r].matmul(torch.diag(s[0:p.r]).matmul(vh[0:p.r,:]))).view(original_shape)

                if hasattr(p,'is_matrix') and p.is_matrix:
                    if not pre:
                        state["pos_l"] = torch.eye(n=n, device=p.device)
                        state["pos_r"] = torch.eye(n=m, device=p.device)
                    grad1 = p.data.view(Dim2_shape).clone()
                    y1_temp = p.u[:,0:p.r].t()@state["pos_l"]
                    y1_part = y1_temp@p.u[:,0:p.r]@y1_temp@grad1
                    y1h = y1_part @ (torch.eye(n=m, device=p.device) - p.vh[0:p.r,:].t()@p.vh[0:p.r,:])
                    y2_temp = state["pos_r"]@p.vh[0:p.r,:].t()
                    y2_part = grad1@y2_temp@p.vh[0:p.r,:]@y2_temp
                    y2 = (torch.eye(n=n, device=p.device) - p.u[:,0:p.r]@p.u[:,0:p.r].t())@y2_part
                    k0 = y1_part@p.vh[0:p.r,:].t() + p.u[:,0:p.r].t()@y2_part - y1_temp@p.u[:,0:p.r]@y1_temp@y2_part

                state["step"] += 1
                # H_r(W_k)
                if hasattr(p,'is_matrix') and p.is_matrix:
                    # Use 2*qr to replace svd
                    q1,k1 = torch.linalg.qr(y1h.t())
                    q2,k2 = torch.linalg.qr(y2)
                    M = torch.cat((torch.cat((k0,k2),0),torch.cat((k1.t(),torch.zeros(k0.size()[0],k0.size()[0],device=p.device)),0)),1)
                    Small = torch.clone(M)
                    u_m, s, vh_m = torch.linalg.svd(Small)
                    u = torch.cat((p.u[:,0:p.r], q2),1)@u_m
                    vh = vh_m@torch.cat((p.vh[0:p.r,:], q1.t()),0)
                    rmax = p.r
                    tmp = 0.0
                    if rmax >= p.minimum_rank: 
                        tol = group["theta"] * torch.linalg.norm(s)
                        rmax = int(np.floor(s.shape[0] / 2))
                        for j in range(0, 2 * rmax - 1):
                            tmp = torch.linalg.norm(s[j:2 * rmax - 1])
                            if tmp < tol:
                                rmax = j
                                break
                        
                        rmax = min([rmax, p.r])
                        rmax = max([rmax, 2])

                        p.s[:rmax] = s[:rmax]
                        p.u[:,:rmax] = u[:,:rmax]
                        p.vh[:rmax,:] = vh[:rmax, :]
                        p.data = (p.u[:,:rmax]@torch.diag(s[:rmax])@p.vh[:rmax,:]).view(original_shape)
                        #ipdb.set_trace()
                        p.r = rmax