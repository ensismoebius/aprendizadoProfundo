import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch import nn, save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from snntorch import utils 
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import snntorch.functional as SF
import matplotlib.pyplot as plt

from data import EEGDataset

class SNNModel(torch.nn.Module):
    def __init__(self, input_timepoints):
        super(SNNModel, self).__init__()

        beta = 0.9  # neuron decay rate 
        spike_grad = surrogate.fast_sigmoid() # fast sigmoid surrogate gradient


        self.inputCurrent = nn.Linear(6, 10)
        self.spike1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.inputSpikes = nn.Linear(10, 27)
        self.spike2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

       
        self.steps = input_timepoints
        self.spk = torch.zeros(1)

    def forward(self, x):
        
        self.mem1 = self.spike1.init_leaky()
        self.mem2 = self.spike2.init_leaky()
        
        for step in range(self.steps):
            inputCurrent = self.inputCurrent(x[:,:,step])
            spk1, self.mem1 = self.spike1(inputCurrent, self.mem1)
            
            inputSpikes = self.inputSpikes(spk1)
            self.spk, self.mem2 = self.spike2(inputSpikes, self.mem2)
            
        return self.spk

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the parameters
batch_size = 4
input_channels = 6  # number of EEG channels
input_timepoints = 4096  # number of time points in each EEG sample
device = torch.device("cpu")

# Create DataLoaders
data_path='/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de Datos Habla Imaginada/S01/S01_EEG.mat'
eegDataset = EEGDataset(data_path)

# Get the indices for the entire dataset
dataset_size = len(eegDataset)
indices = list(range(dataset_size))

# Split the indices into training and validation sets
train_indices, test_indices = train_test_split(indices, test_size=0.8, random_state=42, shuffle=True)

# Create SubsetRandomSamplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders using the SubsetRandomSamplers
batch_size = 4
train_loader = DataLoader(eegDataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(eegDataset, batch_size=batch_size, sampler=test_sampler)

# Create network object
model = SNNModel(input_timepoints)
# Ensures that the networks runs with floats
model.float()
# Initialize lif weights
utils.reset(model)

# Optmizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
loss_fn = nn.CrossEntropyLoss()

# Loss and accuracy history
loss_hist = [] # record loss over iterations 
acc_hist = [] # record accuracy over iterations

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    
    # Retrieve batch
    trainning_batch = iter(train_loader)
    iteration = 0;
    
    for data, targets in trainning_batch:
        
        # More one iteration
        iteration += 1
        
        # Get data and labels
        data = data.to(device)
        targets = targets.to(device)

        # forward-pass
        model.train() # Enable tranning mode on model
        spk_rec = model(data) # The forward pass itself
        
        
        # loss_val = loss_fn(spk_rec, targets.long()) # loss calculation
        loss_val = torch.zeros(1, dtype=torch.float, device=device)
        loss_val += loss_fn(spk_rec.sum(1), targets)
        
        
        # Gradient calculation + weight update
        optimizer.zero_grad() # null gradients
        loss_val.backward() # calculate gradients
        optimizer.step() # update weights
        
        # Store loss history for plotting
        loss_hist.append(loss_val.item()) # store loss

        # print every 25 iterations
        if iteration % 2 == 0:
            print(f"Epoch {epoch}, Iteration {iteration} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            # acc = SF.accuracy_rate(spk_rec.sum(1).detach().unsqueeze(0), targets) 
            # acc_hist.append(acc)
            # print(f"Accuracy: {acc * 100:.2f}%\n")
          
        # uncomment for faster termination
        # if i == 150:
        #     break
