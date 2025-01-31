import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io as io
import time
from tqdm import tqdm
from torch import nn
from torch.nn import init
import shampoo
import network

import torch.nn.functional as F
from collections import OrderedDict

torch.cuda.set_device(0)
dt = time.strftime("%Y-%m-%d %H:%M:%S")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train = torchvision.datasets.MNIST(root='~/Datasets',
                                   train=True, download=True, transform=transform_train)
test = torchvision.datasets.MNIST(root='~/Datasets',
                                  train=False, download=True, transform=transform_test)

batch_size = 128
train_iter = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False, num_workers=0)

# train, val = torch.utils.data.random_split(train, [55000, 5000])

num_epochs = 100

tr_loss = np.zeros(num_epochs + 1)
ts_loss = np.zeros(num_epochs + 1)
vl_loss = np.zeros(num_epochs + 1)
tr_acc = np.zeros(num_epochs + 1)
ts_acc = np.zeros(num_epochs + 1)
vl_acc = np.zeros(num_epochs + 1)
bp_time = np.zeros(num_epochs + 1)


def disable_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def enable_bn(model):
    model.train()


def evaluate_accuracy_loss(data_iter, net):
    acc_sum, loss_sum, n = 0.0, 0.0, 0
    for X, y in data_iter:
        with torch.no_grad():
            X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            loss_sum += loss(y_hat, y)
            n += y.shape[0]
        torch.cuda.empty_cache()
    return acc_sum / n, loss_sum / n


input_size = [1, 28, 28]
net = network.MLP_mnist().cuda()
params = list(net.parameters())
g_list = []
lambda_list = []
w_list = []

# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight, mode='fan_out')
#         if m.bias is not None:
#             init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         init.normal_(m.weight, std=1e-3)  # 1e-3
#         if m.bias is not None:
#             # init.normal_(m.bias, std=1e-3)
#             init.constant_(m.bias, 0)

list_len = len(params)
loss = torch.nn.CrossEntropyLoss()

lr = 0.1
para_weightdecay = 0
momentum = 0
optimizer = shampoo.Shampoo(params=net.parameters(), lr=lr, momentum=momentum,
                            weight_decay=para_weightdecay, shampoo=True)
# tr_acc[0], tr_loss[0] = evaluate_accuracy_loss(train_iter, net)
# ts_acc[0], ts_loss[0] = evaluate_accuracy_loss(test_iter, net)


for epoch in range(num_epochs):
    cal_time = 0
    train_loss, train_acc, n = 0.0, 0.0, 0
    for X, y in tqdm(train_iter):
        # enable_bn(net)
        X = X.cuda()
        y = y.cuda()
        start = time.time()
        g_list = []
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        cal_time = cal_time + end - start
        train_loss += l.item() * y.shape[0]
        train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
        # print(train_loss/n, train_acc/n)
    bp_time[epoch + 1] = cal_time
    disable_bn(net)
    tr_acc[epoch + 1], tr_loss[epoch + 1] = evaluate_accuracy_loss(train_iter, net)
    ts_acc[epoch + 1], ts_loss[epoch + 1] = evaluate_accuracy_loss(test_iter, net)
    enable_bn(net)
    print(epoch + 1, tr_acc[epoch + 1], tr_loss[epoch + 1], ts_acc[epoch + 1], ts_loss[epoch + 1], max(ts_acc),
          bp_time[epoch + 1])
dataSGD = 'data/test.mat'
io.savemat(dataSGD, {'tr_acc': tr_acc, 'tr_loss': tr_loss, 'ts_acc': ts_acc, 'ts_loss': ts_loss, 'vl_acc': vl_acc,
                     'vl_loss': vl_loss, 'epochs': num_epochs,
                     'para_weightdecay': para_weightdecay,
                     'batch': batch_size, 'bp_time': bp_time})
