#%%
# windows
if __name__ == '__main__':
    import torch
    import numpy as np
    import os
    import sys
    import argparse
    from datetime import datetime
    from tqdm import tqdm
    import pandas as pd
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from rgd_fixed_theta import RGD_Opt
    from sklearn.model_selection import train_test_split
    from training_api import *
    from torchvision import datasets, transforms
    from VGG16 import VGG16
    from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print(f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(":::::::", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    parser = argparse.ArgumentParser(description='Pytorch dlrt accuracy vs compression ratio')
    parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                        help='number of epochs for training (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='BATCH_SIZE',
                        help='batch size for training (default: 128)')
    parser.add_argument('--cv_runs', type=int, default=1, metavar='CV_RUNS',
                        help='number of runs for c.i. (default: 10)')
    parser.add_argument('--step', type=float, default=0.1, metavar='STEP',
                        help='step for the timing grid of the experiment (default: 0.1)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate for the training (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0, metavar='MOMENTUM',
                        help='momentum (default: 0.1)')
    args = parser.parse_args()

    MAX_EPOCHS = 400
    current_optimizer = 'RAdaGrad'
    net = 'VGG16'
    if current_optimizer == 'RGD':
        pre = False;
    else:
        pre = True

    def accuracy(outputs, labels):
        return torch.mean(torch.tensor(torch.argmax(outputs.detach(), axis=1) == labels, dtype=torch.float16))

    thetas = [1]  # [0.07,0.09,0.11,0.13]
    lrs = [0.06]
    batches = [64]
    epsilons = [1]


    metric = accuracy
    criterion = torch.nn.CrossEntropyLoss()
    metric_name = 'accuracy'


    def cifar10_augmentation(inputs, training):

        inputs = torch.nn.functional.pad(inputs, (4, 4, 4, 4), mode='reflect')
        if training:

            i, j, h, w = transforms.RandomCrop.get_params(inputs, output_size=(32, 32))
            inputs = transforms.functional.crop(inputs, i, j, h, w)

            if torch.rand(1) < 0.5:
                inputs = transforms.functional.hflip(inputs)


        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(inputs.device)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(inputs.device)
        inputs = (inputs - mean) / std

        return inputs

    def train_lr(NN, optimizer, train_loader, validation_loader, test_loader, criterion, metric, epochs,
                 metric_name='accuracy', device='cpu', count_bias=False, path=None, fine_tune=False, scheduler=None):

        #%%
        running_data = pd.DataFrame(data=None,
                                    columns=['epoch', 'theta', 'learning_rate', 'train_loss', 'train_' + metric_name + '(%)',
                                             'validation_loss', 'validation_' + metric_name + '(%)', 'test_' + metric_name + '(%)',
                                             'ranks', '# effective parameters', 'cr_test (%)', '# effective parameters train',
                                             'cr_train (%)', '# effective parameters train with grads', 'cr_train_grads (%)', 'Time (s)','current_lr'])

        total_params_full = full_count_params(NN, count_bias)
        total_params_full_grads = full_count_params(NN, count_bias, True)

        file_name = path

        if path is not None:
            file_name += '_epoch_' + str(MAX_EPOCHS) + '_.csv'

        # INITIAL VERIFICATION
        with torch.no_grad():
            k = len(validation_loader)
            loss_hist = 0
            acc_hist = 0
            loss_hist_val = 0.0
            acc_hist_val = 0.0
            for i, data in enumerate(validation_loader):  # validation
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = cifar10_augmentation(inputs, training=False)
                outputs = NN(inputs).detach().to(device)
                loss_val = criterion(outputs, labels)
                loss_hist_val += float(loss_val.item()) / k
                acc_hist_val += float(metric(outputs, labels)) / k
                k = len(test_loader)
            print("val_loss:", loss_hist_val)
            loss_hist_test = 0.0
            acc_hist_test = 0.0
            for i, data in enumerate(test_loader):  # validation
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = cifar10_augmentation(inputs, training=False)
                outputs = NN(inputs).detach().to(device)
                loss_test = criterion(outputs, labels)
                loss_hist_test += float(loss_test.item()) / k
                acc_hist_test += float(metric(outputs, labels)) / k
            print('=' * 100)
            ranks = []
            for i, l in enumerate(NN.layer):
                for p in l.parameters():
                    if hasattr(p, 'r'):
                        ranks.append(p.r)
            epoch_data = [0, 0.0, 0.0, round(loss_hist, 3), round(acc_hist * 100, 4), round(loss_hist_val, 3),
                          round(acc_hist_val * 100, 4), round(acc_hist_test * 100, 4), ranks, 0, round(100 * (1 - 0), 4),
                          0, round(100 * (1 - 0), 4), 0, round(100 * (1 - 0), 4), datetime.now(), 'None']
            running_data.loc[0] = epoch_data
            if file_name is not None:
                running_data.to_csv(file_name)
                
        if scheduler is not None:
            scheduler.step()

        for epoch in tqdm(range(epochs)):
            print(f'epoch {epoch}---------------------------------------------')
            loss_hist = 0
            acc_hist = 0
            k = len(train_loader)

            current_lr = optimizer_RGD.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{epochs}] - Current Learning Rate: {current_lr:.8f}")

            for i, data in enumerate(train_loader):  # train
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = cifar10_augmentation(inputs, training=True)


                def closure():
                    loss, outputs = NN.populate_gradients(inputs, labels, criterion)
                    return loss

                optimizer.step(closure=closure)
                optimizer.zero_grad()
                loss, outputs = NN.populate_gradients(inputs, labels, criterion)
                loss = criterion(outputs, labels)
                loss_hist += float(loss.item()) / k

                outputs = outputs.to(device)
                acc_hist += float(metric(outputs, labels)) / k
                torch.cuda.empty_cache()

            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                k = len(validation_loader)
                loss_hist_val = 0.0
                acc_hist_val = 0.0
                for i, data in enumerate(validation_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = NN(inputs).detach().to(device)
                    loss_val = criterion(outputs, labels)
                    loss_hist_val += float(loss_val.item()) / k
                    acc_hist_val += float(metric(outputs, labels)) / k

                k = len(test_loader)
                loss_hist_test = 0.0
                acc_hist_test = 0.0
                for i, data in enumerate(test_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = NN(inputs).detach().to(device)
                    loss_test = criterion(outputs, labels)
                    loss_hist_test += float(loss_test.item()) / k
                    acc_hist_test += float(metric(outputs, labels)) / k
            print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | {metric_name}: {acc_hist:9.4f} | val loss: {loss_hist_val:9.4f} | val {metric_name}:{acc_hist_val:9.4f}')
            print('=' * 100)

            ranks = []
            for i, l in enumerate(NN.layer):
                for p in l.parameters():
                    if hasattr(p, 'r'):
                        print(f'rank layer {i} {p.r}')
                        ranks.append(p.r)
            epoch_data = [epoch, 0.0, 0.0, round(loss_hist, 3), round(acc_hist * 100, 4), round(loss_hist_val, 3),
                          round(acc_hist_val * 100, 4), round(acc_hist_test * 100, 4), ranks, 0, round(100 * (1 - 0), 4),
                          0, round(100 * (1 - 0), 4), 0, round(100 * (1 - 0), 4), datetime.now(), current_lr]
            running_data.loc[epoch + 1] = epoch_data
            if file_name is not None:
                running_data.to_csv(file_name)
            if scheduler is not None:
                scheduler.step()
            if epoch == 0:
                best_val_loss = loss_hist_val
            if loss_hist_val < best_val_loss:
                if hasattr(optimizer, 'theta'):
                    torch.save(NN.state_dict(), path + '_best_weights_' + str(optimizer.theta) + '.pt')
                else:
                    torch.save(NN.state_dict(), path + '_best_weights_base' + '.pt')
        return running_data

    for index, theta in enumerate(thetas, 0):
        for bindex, lr in enumerate(lrs, 0):
            for cindex, batch_size in enumerate(batches, 0):
                for cv_run in range(args.cv_runs):
                    for findex, epsilon in enumerate(epsilons, 0):

                        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),  
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                        ])

                        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                        ])

                        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=transform_train)
                        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                                        transform=transform_test)

                        train_size = int(0.8 * len(train_dataset))
                        val_size = len(train_dataset) - train_size
                        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                                   [train_size, val_size])

                        val_dataset.dataset.transform = transform_test

                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        if net == 'VGG16':
                            f = VGG16(device=device)
                        f = f.to(device)
                        optimizer_RGD = RGD_Opt(f, lr=lr, momentum=args.momentum, weight_decay=5e-5, epsilon=epsilon,
                                                theta=theta, pre=pre)

                        if net == 'VGG16':
                            path = './results_RAdaGrad_VGG16/_My_running_data_net_' + str(net) + '_theta_' + str(
                                theta) + '_' + str(current_optimizer) + '_lr_' + str(lr) + '_batch_' + str(
                                batch_size) + '_cifar_'  + "_epsilon_" + str(epsilon)

                        print('=' * 100)
                        print(f'run number {index} \n theta = {theta} \n lr = {lr}')
                        train_results = train_lr(f, optimizer_RGD, train_loader, val_loader, test_loader, criterion,
                                                 metric, MAX_EPOCHS, metric_name=metric_name, device=device,
                                                 count_bias=False, path=path)