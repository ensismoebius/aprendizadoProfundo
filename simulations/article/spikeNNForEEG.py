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
    def __init__(self, inputLength, insist = 4, neuronDecayRate = 0.9):
        super(SNNModel, self).__init__()

        self.insist = insist
        spike_grad = surrogate.fast_sigmoid() # fast sigmoid surrogate gradient

        self.inputCurrent = nn.Linear(inputLength, 10000)
        self.spike1 = snn.Leaky(beta=neuronDecayRate, spike_grad=spike_grad)
        self.inputSpikes = nn.Linear(10000, 27)
        self.spike2 = snn.Leaky(beta=neuronDecayRate, spike_grad=spike_grad)
        self.avg = nn.AvgPool1d(27, stride=1)


    def forward(self, x):
        mem1 = self.spike1.init_leaky()
        mem2 = self.spike2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []
        
        for step in range(self.insist):
            inputCurrent = self.inputCurrent(x)
            spk1, mem1 = self.spike1(inputCurrent, mem1)
            
            inputSpikes = self.inputSpikes(spk1)
            spk2, mem2 = self.spike2(inputSpikes, mem2)
            
            avg = self.avg(spk2)
            
            spk2_rec.append(avg)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
# Configuration section
insist = 4
test_size = 0.1
batch_size = 10
num_epochs = 100
neuronDecayRate = 0.9
device = torch.device("cuda")
data_path='/content/drive/MyDrive/databases/S01_EEG.mat'
# device = torch.device("cpu")
# data_path='/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de Datos Habla Imaginada/S01/S01_EEG.mat'


# Set random seed for reproducibility
torch.manual_seed(42)

# Create DataLoaders
eegDataset = EEGDataset(data_path)

# Get the indices for the entire dataset
dataset_size = len(eegDataset)
indices = list(range(dataset_size))

# Split the indices into training and validation sets
train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)

# Create SubsetRandomSamplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders using the SubsetRandomSamplers
train_loader = DataLoader(eegDataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(eegDataset, batch_size=batch_size, sampler=test_sampler)

# Create network object
model = SNNModel(
    inputLength=eegDataset.channelsLength, 
    insist=insist, 
    neuronDecayRate=neuronDecayRate
)
model.to(device)
# Ensures that the networks runs with floats
model.float()
# Initialize lif weights
utils.reset(model)

# Optmizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.1, 0.999))
loss_fn = nn.CrossEntropyLoss()

# Loss and accuracy history
min_loss_hist = [] # record loss over iterations 

# Keeps the minimum loss
minLoss = 1000000

# Keeps the maximum loss
maxLoss = 0

# Training loop
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
        spk_rec, _ = model(data) # The forward pass itself
        
        loss_val = torch.zeros(1, dtype=torch.float, device=device)
        loss_val += loss_fn(spk_rec.sum(2).sum(0).squeeze(), targets)
        
        # Gradient calculation + weight update
        optimizer.zero_grad() # null gradients
        loss_val.backward() # calculate gradients
        optimizer.step() # update weights
        
        # Keeps track of the max and min loss
        if maxLoss < loss_val.item() :
            maxLoss = loss_val.item()
        if minLoss > loss_val.item() :
            minLoss = loss_val.item()
            min_loss_hist.append(minLoss) # store loss

        # print every 10 iterations
        if iteration % 2 == 0:
            print(f"Epoch {epoch}, Iteration {iteration} \nTrain max/min loss: {maxLoss:.2f}/{minLoss:.2f}")

    plt.plot(min_loss_hist)
    
model.save_model('spik.pth')
# model.load_model('test.pth')