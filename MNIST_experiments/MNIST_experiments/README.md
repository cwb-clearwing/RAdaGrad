
# MNIST-FC & LeNet-5

## Description

This project classifies the MNIST dataset using two neural network architectures:

- **Fully Connected Networks (FC)**
- **LeNet-5**

The training process is performed with three different optimizers:

- **RAdaGrad**
- **RGD**
- **DLRT**

---

## Files

- `train_RAdaGrad_RGD_FC_leNet5.py`:  
  Experiments with Fully Connected Networks (FC) or LeNet-5 using the RAdaGrad or RGD optimizers.  
  You can specify the network and optimizer by modifying lines 40 and 41 in the script.

- `train_DLRT_FC.py`:  
  Experiments with Fully Connected Networks (FC) using the DLRT optimizer.

- `train_DLRT_leNet5.py`:  
  Experiments with LeNet-5 using the DLRT optimizer.

---

## Results

The results of the experiments are saved in the following directories:

- `results_Full_Connection`:  
  Contains results for all the optimizers applied to Fully Connected Networks (FC).

- `results_leNet-5`:  
  Contains results for all the optimizers applied to LeNet-5.

Each folder contains the output data and performance metrics for the corresponding network and optimizer.
