#%%
# windows
if __name__ ==  '__main__':
  import torch
  import numpy as np
  import os 
  import sys 
  import argparse
  sys.path.insert(1, os.path.join(sys.path[0], '..'))
  from optimizer_KLS_DLRT.dlrt_optimizer import dlr_opt
  from optimizer_KLS_DLRT.train_custom_optimizer import *
  import tensorflow as tf
  from models_folder_DLRT.Lenet5 import Lenet5
  from sklearn.model_selection import train_test_split
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")
  print(f"{torch.cuda.get_device_name(torch.cuda.current_device())}")

  parser = argparse.ArgumentParser(description='Pytorch dlrt accuracy vs compression ratio')  
  parser.add_argument('--epochs', type=int, default=20, metavar='EPOCHS',
                      help='number of epochs for training (default: 100)')  
  parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                      help='batch size for training (default: 128)')  
  parser.add_argument('--cv_runs', type=int, default=1, metavar='CV_RUNS',
                      help='number of runs for c.i. (default: 10)')  
  parser.add_argument('--step', type=float, default=0.1, metavar='STEP',
                      help='step for the timing grid of the experiment (default: 0.1)')
  parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                      help='learning rate for the training (default: 0.05)')
  parser.add_argument('--momentum', type=float, default=0.1, metavar='MOMENTUM',
                      help='momentum (default: 0.1)')                                                 
  args = parser.parse_args()


  MAX_EPOCHS = 10


  def accuracy(outputs,labels):

      return torch.mean(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = float16))


  thetas = [0.15]#[0.07,0.09,0.11,0.13]
  lrs = [0.03]
  batches = [128]

  metric  = accuracy
  criterion = torch.nn.CrossEntropyLoss()
  metric_name = 'accuracy'

  for index,theta in enumerate(thetas,0):
    for bindex,lr in enumerate(lrs,0):
      for cindex,batch_size in enumerate(batches,0):      
        for cv_run in range(args.cv_runs):
          (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
          x = np.vstack([x_train,x_test])
          y = np.hstack([y_train,y_test])
          x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=60000,stratify = y)
          ## for cifar
          x_train,x_test = x_train.reshape(x_train.shape[0],1,x_train.shape[1],x_train.shape[2]),x_test.reshape(x_test.shape[0],1,x_test.shape[1],x_test.shape[2])  
          y_train,y_test = y_train.reshape(y_train.shape[0]),y_test.reshape(y_test.shape[0])
          ##
          x_train,x_test,y_train,y_test = torch.tensor(x_train).float()/255,torch.tensor(x_test).float()/255,torch.tensor(y_train),torch.tensor(y_test)
          x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 50000,stratify = y_train)
          ##
          print(f'train shape {x_train.shape}')
          print(f'val shape {x_val.shape}')
          print(f'test shape {x_test.shape}')
          batch_size_train,batch_size_test = batch_size,batch_size
          train_loader = torch.utils.data.DataLoader(
            [(x_train[i],y_train[i]) for i in range(x_train.shape[0])],
            batch_size=batch_size_train, shuffle=True)
          val_loader = torch.utils.data.DataLoader(
            [(x_val[i],y_val[i]) for i in range(x_val.shape[0])],
            batch_size=batch_size_test, shuffle=True)
          test_loader = torch.utils.data.DataLoader(
            [(x_test[i],y_test[i]) for i in range(x_test.shape[0])],
            batch_size=batch_size_test, shuffle=True)
          f = Lenet5(device = device)
          f = f.to(device)
          optimizer = dlr_opt(f,tau = lr,theta = theta,KLS_optim=torch.optim.SGD)
          optimizer_base = torch.optim.SGD(f.parameters(),lr = args.lr,momentum=args.momentum)
          # path = './results_Lenet5/_My_running_data_'+str(optimizer.theta)+'_'+str(cv_run)
          path = './results_lenet-5/_My_running_data_net_leNet-5'+'_theta_'+str(theta)+'_dlrt_lr_'+str(lr)+'_batch_'+str(batch_size)
          print('='*100)
          print(f'run number {index} \n theta = {theta}')
          #try:
          train_results = train_dlrt(f,optimizer,train_loader,val_loader,test_loader,criterion,\
                                      metric,MAX_EPOCHS,metric_name = metric_name,device = device,count_bias = False,path = path)