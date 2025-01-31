import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import torch.nn.init as init
import math 

class LowRank_Linear(nn.Module):
    def __init__(self, in_features, out_features, rank, r_min, device):
        super(LowRank_Linear, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.layer.weight.is_matrix = True
        self.rmax = int(min([self.in_features, self.out_features]) / 2)
        self.layer.weight.r = min([rank,self.rmax])
        self.layer.weight.minimum_rank=r_min
        self.layer.weight.s = None
        self.layer.weight.u = None
        self.layer.weight.vh = None
        self.layer.bias.is_matrix = False
        self.device = device
        self.lr = True
        self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.layer.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        #self.weight.r = self.rank
        #self.weight.is_matrix = True

    def forward(self, x):
        y = self.layer(x)
        return y


class Shampoo(Optimizer):

    def __init__(self, params, lr=1e-1, momentum=0, weight_decay=0, shampoo=False):
        """

        :param params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param weight_decay: (float, optional): weigt decay factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional): update frequency to compute inverse (default: 1)
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, shampoo=shampoo)
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                shampoo = group["shampoo"]
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    if p.is_matrix:
                        if shampoo:
                            state["l_buffer"] = torch.diag(torch.eye(n=p.size()[0], device=p.device))
                            state["r_buffer"] = torch.diag(torch.eye(n=p.size()[1], device=p.device))
                        state["svd"] = torch.linalg.svd(p.data)
                    # for dim_id, dim in enumerate(grad.size()):
                    #     # precondition matrices
                    #     state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                    #     state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    # grad = (1 - moment) * grad(t) + moment * grad(t-1)
                    # and grad(-1) = grad(0)
                    grad.mul_(1 - momentum).add_(momentum, state["momentum_buffer"])

                if weight_decay > 0:
                    grad.add_(group["weight_decay"], p.data)

                if shampoo and p.is_matrix:
                    temp = grad.detach().clone()
                    state["l_buffer"] += torch.diag(temp.matmul(temp.t()))
                    state["r_buffer"] += torch.diag(temp.t().matmul(temp))
                    t1 = torch.diag(state["l_buffer"] ** (-1 / 4))
                    t2 = torch.diag(state["r_buffer"] ** (-1 / 4))
                    grad = t1.matmul(grad.matmul(t2))

                # SVD:
                if p.is_matrix:
                    u, s, vh = state["svd"]
                    grad = (u.matmul(u.t())).matmul(grad) + grad.matmul(vh.t().matmul(vh)) - (u.matmul(u.t())).matmul(
                        grad.matmul(vh.t().matmul(vh)))

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(-group["lr"], grad)

                if p.is_matrix:
                    state["svd"] = torch.linalg.svd(p.data)
                    u, s, vh = state["svd"]
                    s = s[0:p.r]
                    s = torch.nn.functional.pad(torch.diag(s), [0, vh.size()[0] - p.r, 0, u.size()[1] - p.r])
                    p.data = u.matmul(s.matmul(vh))

        return loss
