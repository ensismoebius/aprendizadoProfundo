import math
import torch
import torch.nn as nn
from data import EEGDataset
from torch import nn, save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

import matplotlib.pyplot as plt

# Define a simple neural network architecture with dynamic size calculation
class EEGNet(nn.Module):
    def __init__(self, input_channels, input_timepoints):
        super(EEGNet, self).__init__()
        
        ############################################################
        ######## Calculates the size of the feature vector #########
        ############################################################
        
        # Calculate the size of the features after the 1st conv and pool layers
        feature_size = self.get_tensor_size_after_conv(input_timepoints,5)
        feature_size = self.get_tensor_size_after_maxpool(feature_size,2)
        # Calculate the size of the features after the 2nd conv and pool layers
        feature_size = self.get_tensor_size_after_conv(feature_size,5)
        feature_size = self.get_tensor_size_after_maxpool(feature_size,2)
        # Calculate the size of the features after the flatten
        feature_size = feature_size * 32
        
        
        ################################
        ###### Creates the model #######
        ################################
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 27)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    def start_train(self, dataloader, epochs = 10, lr=0.0001):

        # Define a loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 6 == 0:  # print every 6 mini-batches
                    print('[%d, %5d] loss: %.8f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')
        
    def save_model(self, path):
        with open(path, 'wb') as f: 
            save(self.state_dict(), f)
            
    def load_model(self, path):
        with open(path, 'rb') as f: 
            self.load_state_dict(load(f))  
    
    def get_tensor_size_after_conv(self, input_size, convolution_kernel_size):
        return 1 +(input_size - convolution_kernel_size)

    def get_tensor_size_after_maxpool(self, input_size, poll_kernel_size):
        return math.floor(input_size / poll_kernel_size)

# Create the neural network
test_size = 0.1
batch_size = 10
num_epochs = 100
input_channels = 6  # number of EEG channels
input_timepoints = 4096  # number of time points in each EEG sample
# device = torch.device("cuda")
# data_path='/content/drive/MyDrive/databases/S01_EEG.mat'
device = torch.device("cpu")
data_path='/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de Datos Habla Imaginada/S01/S01_EEG.mat'

model = EEGNet(input_channels, input_timepoints)

# Optmizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.1, 0.999))
loss_fn = nn.CrossEntropyLoss()

# Ensures that the networks runs with floats
model.float()

# Create DataLoaders
eegDataset = EEGDataset(data_path)

# Get the indices for the entire dataset
dataset_size = len(eegDataset)
indices = list(range(dataset_size))

# Split the indices into training and validation sets
train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

# Create SubsetRandomSamplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders using the SubsetRandomSamplers
train_loader = DataLoader(eegDataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(eegDataset, batch_size=batch_size, sampler=test_sampler)


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
        spk_rec = model(data) # The forward pass itself
        
        loss_val = torch.zeros(1, dtype=torch.float, device=device)
        loss_val += loss_fn(spk_rec.sum(1).squeeze(), targets)
        
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

# model.save_model('test.pth')
# model.load_model('test.pth')