import torch
from torch import nn
from torch.optim.optimizer import Optimizer

class RGD_Opt(Optimizer):
    def __init__(self, params, lr=1e-1, momentum=0, weight_decay=0,epsilon=0.01):
        """

        :param params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param weight_decay: (float, optional): weigt decay factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional): update frequency to compute inverse (default: 1)
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(RGD_Opt, self).__init__(params, defaults)

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
                # Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    if hasattr(p,'is_matrix') and p.is_matrix:
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

                if hasattr(p,'is_matrix') and p.is_matrix:
                    temp = grad.detach().clone()
                    state["l_buffer"] = torch.eye(n=p.size()[0], device=p.device).mul(group["epsilon"]) + temp.matmul(temp.t())
                    state["r_buffer"] = torch.eye(n=p.size()[1], device=p.device).mul(group["epsilon"]) + temp.t().matmul(temp)
                    t1 = torch.diag(state["l_buffer"] ** (-1 / 4))
                    t2 = torch.diag(state["r_buffer"] ** (-1 / 4))
                    grad = t1.matmul(grad.matmul(t2))
                    # SVD:
                    u, s, vh = state["svd"]
                    grad = (u.matmul(u.t())).matmul(grad) + grad.matmul(vh.t().matmul(vh)) - (u.matmul(u.t())).matmul(
                        grad.matmul(vh.t().matmul(vh)))

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(-group["lr"], grad)

                # H_r(X_k)
                if hasattr(p,'is_matrix') and p.is_matrix:
                    state["svd"] = torch.linalg.svd(p.data)
                    u, s, vh = state["svd"]
                    s = s[0:p.r]
                    s = torch.nn.functional.pad(torch.diag(s), [0, vh.size()[0] - p.r, 0, u.size()[1] - p.r])
                    p.data = u.matmul(s.matmul(vh))

        return loss