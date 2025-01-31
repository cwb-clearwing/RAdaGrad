#%%
# windows
if __name__ ==  '__main__':
  import torch
  import numpy as np
  import os


  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  import sys 
  import argparse
  sys.path.insert(1, os.path.join(sys.path[0], '..'))
  from optimizer_KLS_DLRT.dlrt_optimizer import dlr_opt
  from optimizer_KLS_DLRT.train_custom_optimizer_aug import *
  import tensorflow as tf
  from models_folder_DLRT.VGG16 import VGG16
  from sklearn.model_selection import train_test_split
  from torchvision import datasets, transforms

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")
  print(f"{torch.cuda.get_device_name(torch.cuda.current_device())}")

  parser = argparse.ArgumentParser(description='Pytorch dlrt accuracy vs compression ratio')  
  parser.add_argument('--epochs', type=int, default=1000, metavar='EPOCHS',
                      help='number of epochs for training (default: 100)')  
  parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                      help='batch size for training (default: 128)')  
  parser.add_argument('--cv_runs', type=int, default=1, metavar='CV_RUNS',
                      help='number of runs for c.i. (default: 10)')  
  parser.add_argument('--step', type=float, default=0.1, metavar='STEP',
                      help='step for the timing grid of the experiment (default: 0.1)')
  parser.add_argument('--lr', type=float, default=0.04, metavar='LR',
                      help='learning rate for the training (default: 0.05)')
  parser.add_argument('--momentum', type=float, default=0, metavar='MOMENTUM',
                      help='momentum (default: 0.1)')                                                 
  args = parser.parse_args()


  MAX_EPOCHS = args.epochs
  current_optimizer = 'DLRT'


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

  def accuracy(outputs,labels):

      return torch.mean(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = float16))



  thetas = [0.2]#[0.07,0.09,0.11,0.13]
  lrs = [0.035]
  batch_sizes = [128]


  metric  = accuracy
  criterion = torch.nn.CrossEntropyLoss() 
  metric_name = 'accuracy'



  for index,theta in enumerate(thetas,0):


    for cv_run in range(args.cv_runs):

      for bindex, lr in enumerate(lrs,0):
            for dindex, batch_size in enumerate(batch_sizes,0):

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

              train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
              test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

              train_size = int(0.8 * len(train_dataset))
              val_size = len(train_dataset) - train_size
              train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

              val_dataset.dataset.transform = transform_test


              batch_size_train,batch_size_test = batch_size,batch_size

              train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
              val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
              test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


              f = VGG16(device = device)
              f = f.to(device)
              optimizer = dlr_opt(f,tau = lr,theta = theta,KLS_optim=torch.optim.SGD)
              path = './results_DLRT_VGG16/_My_running_data_'+str(optimizer.theta)+'_'+str(current_optimizer)+'_'+str(cv_run)+'_batch_'+str(batch_size)+'_lr_'+str(lr)+'_epoch_'+str(MAX_EPOCHS)

              print('='*100)
              print(f'run number {index} \n theta = {theta}')
              #try:

              if current_optimizer == 'DLRT':
                train_results = train_dlrt(f,optimizer,train_loader,val_loader,test_loader,criterion,\
                                                  metric,MAX_EPOCHS,metric_name = metric_name,device = device,count_bias = False,path = path)
              #except Exception as e:
              #  print(e)
              #  print('training went bad')




