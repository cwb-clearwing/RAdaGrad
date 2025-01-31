#%%
# windows
if __name__ ==  '__main__':
  import torch
  import numpy as np
  import os 
  import sys 
  import argparse
  import cProfile
  sys.path.insert(1, os.path.join(sys.path[0], '..'))
  from rgd_fixed_theta import RGD_Opt
  import tensorflow as tf
  from network_MNIST import FC,Lenet5
  from sklearn.model_selection import train_test_split
  from training_api import *
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")
  print(f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
  print(":::::::",os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


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


  MAX_EPOCHS = 200
  net = "leNet-5"  # leNet-5 or FC
  current_optimizer = 'RAdaGrad'
  #net = 'full-connection'
  if current_optimizer == 'RGD':
    pre=False
  else:
    pre=True

  def accuracy(outputs,labels):

      return torch.mean(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = float16))

  thetas = [0.5]#[0.07,0.09,0.11,0.13]
  lrs = [0.07]
  batches = [128]

  metric  = accuracy
  criterion = torch.nn.CrossEntropyLoss() 
  metric_name = 'accuracy'

  def train_lr(NN,optimizer,train_loader,validation_loader,test_loader,criterion,metric,epochs,
                metric_name = 'accuracy',device = 'cpu',count_bias = False,path = None,fine_tune = False,scheduler = None):
    #%%
    running_data = pd.DataFrame(data = None,columns = ['epoch','theta','learning_rate','train_loss','train_'+metric_name+'(%)','validation_loss',\
                                                        'validation_'+metric_name+'(%)','test_'+metric_name+'(%)',\
                                                     'ranks','# effective parameters','cr_test (%)','# effective parameters train','cr_train (%)',\
                                                     '# effective parameters train with grads','cr_train_grads (%)','Time (s)'])

    total_params_full = full_count_params(NN,count_bias)
    total_params_full_grads = full_count_params(NN,count_bias,True)
    #scheduler_rate = optimizer.scheduler_change_rate

    file_name = path

    if path is not None:
      file_name += '_epoch_'+str(MAX_EPOCHS)+'_.csv'#'\_running_data_'+str(optimizer.theta)+'.csv'
    # INITIAL VERIFICATION
    with torch.no_grad():
      k = len(validation_loader)
      loss_hist = 0
      acc_hist = 0
      loss_hist_val = 0.0
      acc_hist_val = 0.0
      for i,data in enumerate(validation_loader):# validation
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = NN(inputs).detach().to(device)
        loss_val = criterion(outputs,labels)
        loss_hist_val+=float(loss_val.item())/k
        acc_hist_val += float(metric(outputs,labels))/k
      k = len(test_loader)
      loss_hist_test = 0.0
      acc_hist_test = 0.0
      for i, data in enumerate(test_loader):  # validation
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = NN(inputs).detach().to(device)
        loss_test = criterion(outputs, labels)
        loss_hist_test += float(loss_test.item()) / k
        acc_hist_test += float(metric(outputs, labels)) / k
      print('='*100)
      ranks = []
      for i,l in enumerate(NN.layer):
        for p in l.parameters():
          if hasattr(p,'r'):
              print(f'rank layer {i} {p.r}')
              ranks.append(p.r)
      print('\n')
      epoch_data = [0,0.0,0.0,round(loss_hist,3),round(acc_hist*100,4),round(loss_hist_val,3),\
                  round(acc_hist_val*100,4),round(acc_hist_test*100,4),ranks,0,round(100*(1-0),4),\
                      0,round(100*(1-0),4),0,round(100*(1-0),4),datetime.now()]
      running_data.loc[0] = epoch_data
      if file_name is not None:
        running_data.to_csv(file_name)
        if scheduler is not None:
            scheduler.step(loss_hist)

    for epoch in tqdm(range(epochs)):
      print(f'epoch {epoch}---------------------------------------------')
      loss_hist = 0
      acc_hist = 0
      k = len(train_loader)
      for i,data in enumerate(train_loader):  # train
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        def closure():
          #loss = NN.populate_gradients(inputs,labels,criterion,step = 'S')
          loss,outputs = NN.populate_gradients(inputs,labels,criterion)
          return loss
        #cProfile.runctx('optimizer.step(closure = closure)',globals=None,locals={'optimizer':optimizer,'closure':closure})
        optimizer.step(closure = closure)
        optimizer.zero_grad()
        loss,outputs = NN.populate_gradients(inputs,labels,criterion)
        loss_hist+=float(loss.item())/k
        outputs = outputs.to(device)#NN(inputs).detach().to(device)
        acc_hist += float(metric(outputs,labels))/k
        #print(f'iter[{i}] | datatime[{datetime.now()}]')
        #ranks=[]
        #for il,l in enumerate(NN.layer):
        #  for p in l.parameters():
        #    if hasattr(p,'r'):
        #      ranks.append(p.r)
        #print(f'rank layer: {ranks}')
      print("Start Validation")
      with torch.no_grad():
        k = len(validation_loader)
        loss_hist_val = 0.0
        acc_hist_val = 0.0
        for i,data in enumerate(validation_loader):# validation
          inputs,labels = data
          inputs,labels = inputs.to(device),labels.to(device)
          outputs = NN(inputs).detach().to(device)
          loss_val = criterion(outputs,labels)
          loss_hist_val+=float(loss_val.item())/k
          acc_hist_val += float(metric(outputs,labels))/k

        k = len(test_loader)
        loss_hist_test = 0.0
        acc_hist_test = 0.0
        for i, data in enumerate(test_loader):  # validation
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = NN(inputs).detach().to(device)
          loss_test = criterion(outputs, labels)
          loss_hist_test += float(loss_test.item()) / k
          acc_hist_test += float(metric(outputs, labels)) / k
      print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | {metric_name}: {acc_hist:9.4f} | val loss: {loss_hist_val:9.4f} | val {metric_name}:{acc_hist_val:9.4f}')
      print('='*100)
      ranks = []
      for i,l in enumerate(NN.layer):
        for p in l.parameters():
          if hasattr(p,'r'):
              #for p in l.parameters():
                  #print(p.data, p.shape, p.r)
              print(f'rank layer {i} {p.r}')
              ranks.append(p.r)
      print('\n')
      #params_test = count_params_test(NN,count_bias)
      #cr_test = round(params_test/total_params_full,3)
      #params_train = count_params_train(NN,count_bias)
      #cr_train = round(params_train/total_params_full,3)
      #params_train_grads = count_params_train(NN,count_bias,True)
      #cr_train_grads = round(params_train_grads/total_params_full_grads,3)
      epoch_data = [epoch,0.0,0.0,round(loss_hist,3),round(acc_hist*100,4),round(loss_hist_val,3),\
                  round(acc_hist_val*100,4),round(acc_hist_test*100,4),ranks,0,round(100*(1-0),4),\
                      0,round(100*(1-0),4),0,round(100*(1-0),4),datetime.now()]
      running_data.loc[epoch+1] = epoch_data
      #if file_name is not None and (epoch%10 == 0 or epoch == epochs-1):
      if file_name is not None:
        running_data.to_csv(file_name)
        if scheduler is not None:
            scheduler.step(loss_hist)
        # if epoch%scheduler_rate == 0:

        #     optimizer.scheduler_step()
      if epoch == 0:
        best_val_loss = loss_hist_val
      if loss_hist_val<best_val_loss:
        if hasattr(optimizer,'theta'):
          torch.save(NN.state_dict(),path+'_best_weights_'+str(optimizer.theta)+'.pt')
        else:
          torch.save(NN.state_dict(),path+'_best_weights_base'+'.pt')
    return running_data

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
          x_train,x_test,y_train,y_test = torch.tensor(x_train).float()/255,torch.tensor(x_test).float()/255,torch.tensor(y_train),torch.tensor(y_test)
          x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 50000,stratify = y_train)
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
          if net == 'leNet-5':
            f = Lenet5(device = device)
          else:
            f = FC(device = device)
          f = f.to(device)
          optimizer_RGD = RGD_Opt(f,lr = lr,momentum=args.momentum,weight_decay = 0,epsilon = 1,theta=theta,pre=pre)
          if net == 'FC':
            path = './results_Full_Connection/_My_running_data_net_'+str(net)+'_theta_'+str(theta)+'_'+str(current_optimizer)+'_lr_'+str(lr)+'_batch_'+str(batch_size)
          elif net == 'leNet-5':
            path = './results_leNet-5/_My_running_data_net_'+str(net)+'_theta_'+str(theta)+'_'+str(current_optimizer)+'_lr_'+str(lr)+'_batch_'+str(batch_size)
          print('='*100)
          print(f'run number {index} \n theta = {theta} \n lr = {lr}')
          #try:
          train_results = train_lr(f,optimizer_RGD,train_loader,val_loader,test_loader,criterion,\
                                              metric,MAX_EPOCHS,metric_name = metric_name,device = device,count_bias = False,path = path)