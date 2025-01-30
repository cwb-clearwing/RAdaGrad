from torch.optim.optimizer import Optimizer, required
import torch
import math
import copy
import time
from torch import nn
import numpy as np
import time
import ipdb

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

class SGDr(Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.98), eps=1e-6, correct_bias=True, reg=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.reg = reg 
        print(f'{self.reg=}')
    def step(self, closure=None):
        for group in self.param_groups:
            for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                grad1 = p1.grad.data
                scale1 = p2.data
                try:
                    grad1_scaled = torch.inverse(scale1.T@scale1+self.reg*torch.eye(scale1.shape[1]).to(scale1.device))@grad1
                except:
                    grad1_scaled = grad1
                
                grad2 = p2.grad.data
                scale2 = p1.data
                try:
                    grad2_scaled = grad2@torch.inverse(scale2@scale2.T+self.reg*torch.eye(scale2.shape[0]).to(scale2.device))
                except:
                    grad2_scaled = grad2
                
                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])
                    p2.data.add_(p2.data, alpha=-group["lr"] * group["weight_decay"])

                p1.data.add_(grad1_scaled, alpha=-group['lr'])
                p2.data.add_(grad2_scaled, alpha=-group['lr'])

class SGD(Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.98), eps=1e-6, correct_bias=True):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                p.data.add_(p.grad.data, alpha=-group['lr'])
            

class AdamWr(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=False, reg=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.reg = reg
        print(f'{self.reg=}')
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                state = self.state[p1]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p1.data)
                    state["exp_avg_sq"] = torch.zeros_like(p1.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                grad1 = p1.grad.data
                c = p2.data
                try:
                    c_ = torch.inverse(c.T@c+self.reg*torch.eye(c.shape[1]).to(c.device))
                except:
                    c_ = torch.eye((c.T@c).shape[0]).to(c.device)
                grad1_scaled = c_@grad1
                assert grad1_scaled.shape == p1.grad.data.shape

                exp_avg.mul_(beta1).add_(grad1_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad1_scaled, grad1_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                c1 = p1.data

                p1.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])

                
                state = self.state[p2]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p2.data)
                    state["exp_avg_sq"] = torch.zeros_like(p2.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                grad2 = p2.grad.data
                try:
                    c1_ = torch.inverse(c1@c1.T+self.reg*torch.eye(c1.shape[0]).to(c1.device))
                except:
                    c1_ = torch.eye((c1@c1.T).shape[0]).to(c1.device)
                
                grad2_scaled = grad2@c1_
                assert grad2_scaled.shape == p2.grad.data.shape
                
                exp_avg.mul_(beta1).add_(grad2_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad2_scaled, grad2_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1


                p2.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p2.data.add_(p2.data, alpha=-group["lr"] * group["weight_decay"])
                
        return loss

class RGD_Opt(Optimizer):
    def __init__(self, params, lr=1e-1, betas=(0.9, 0.98), momentum=0, weight_decay=0.0,epsilon=1,theta=0.1,eps=1e-6,pre=False):
        """

        :param params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param weight_decay: (float, optional): weigt decay factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional): update frequency to compute inverse (default: 1)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, momentum=momentum, weight_decay=weight_decay,epsilon=epsilon,timer1=0,timer2=0,theta=theta,pre=pre)
        super(RGD_Opt, self).__init__(params, defaults)                                                                                                                                                                                                                                                     
    
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def integration_step(self):                                                                                                                             
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Rgd does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

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