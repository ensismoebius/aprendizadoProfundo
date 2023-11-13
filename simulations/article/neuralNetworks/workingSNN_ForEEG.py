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

class SNNModel(nn.Module):
    def __init__(self, inputLength, insist = 4, neuronDecayRate = 0.9):
        super(SNNModel, self).__init__()

        self.insist = insist
        spike_grad = surrogate.fast_sigmoid() # fast sigmoid surrogate gradient

        ################################
        ###### Creates the model #######
        ################################
        
        self.model = nn.Sequential(
            nn.Linear(inputLength, 10000),
            snn.Leaky(beta=neuronDecayRate, spike_grad=spike_grad),
            nn.Linear(10000, 27),
            snn.Leaky(beta=neuronDecayRate, spike_grad=spike_grad),
            nn.AvgPool1d(27, stride=1)
        )
        
    def save_model(self, path):
        with open(path, 'wb') as f: 
            save(self.state_dict(), f)
            
    def load_model(self, path):
        with open(path, 'rb') as f: 
            self.load_state_dict(load(f)) 


    def forward(self, x):
        mem1 = self.model[1].init_leaky()
        mem2 = self.model[3].init_leaky()
        
        spk2_rec = []
        mem2_rec = []
        
        for step in range(self.insist):
            inputCurrent = self.model[0](x)
            spk1, mem1 = self.model[1](inputCurrent, mem1)
            
            inputSpikes = self.model[2](spk1)
            spk2, mem2 = self.model[3](inputSpikes, mem2)
            
            avg = self.model[4](spk2)
            
            spk2_rec.append(avg)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
# Configuration section
insist = 4
test_size = 0.2
batch_size = 10
num_epochs = 100
neuronDecayRate = 0.9
# device = torch.device("cuda")
# data_path='/content/drive/MyDrive/databases/S01_EEG.mat'
device = torch.device("cpu")
data_path='S01_EEG.mat'


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