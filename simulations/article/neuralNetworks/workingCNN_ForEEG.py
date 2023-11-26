import math
import torch
import torch.nn as nn
from data import EEGDataset
from torch import nn, save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
test_size = 0.2
batch_size = 10
num_epochs = 100
# device = torch.device("cuda")
device = torch.device("cpu")

# Load and preprocess data
# data_path='/content/drive/MyDrive/databases/S01_EEG.mat'
data_path='/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de Datos Habla Imaginada/S01/S01_EEG.mat'
eegDataset = EEGDataset(data_path)

# Create model
model = EEGNet(eegDataset.channels, eegDataset.channelsLength)

# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)

# Optmizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.1, 0.999))
loss_fn = nn.CrossEntropyLoss()

# Ensures that the networks runs with floats
model.float()

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
accuracy_hist = []  # record accuracy over iterations

# Keeps the minimum loss
min_loss = 1000000

# Keeps the maximum loss
max_loss = 0

# Training loop
for epoch in range(num_epochs):
    # Initialize variables for computing accuracy and confusion matrix
    correct_preds = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        # forward-pass
        model.train()
        prediction = model(data)

        loss_val = torch.zeros(1, dtype=torch.float, device=device)
        loss_val += loss_fn(prediction.sum(1).squeeze(), targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Compute accuracy and update confusion matrix
        _, predicted = torch.max(prediction.data, 1)
        correct_preds += (predicted == targets).sum().item()
        total_samples += targets.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

        # Keeps track of the max and min loss
        if max_loss < loss_val.item():
            max_loss = loss_val.item()
        if min_loss > loss_val.item():
            min_loss = loss_val.item()

    # Calculate accuracy and loss
    accuracy = correct_preds / total_samples
    avg_loss = loss_val.item() / len(train_loader)

    # Update history
    accuracy_hist.append(accuracy)
    min_loss_hist.append(avg_loss)

    # Print and store results every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Accuracy: {accuracy:.2%}, Train Loss: {avg_loss:.4f}")

# Plot average accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(accuracy_hist, label='Accuracy')
plt.title('Average Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(min_loss_hist, label='Loss')
plt.title('Average Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate on the test set
model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for test_data, test_targets in test_loader:
        test_data, test_targets = test_data.to(device), test_targets.to(device)
        test_outputs = model(test_data)
        _, test_preds = torch.max(test_outputs.data, 1)
        all_test_preds.extend(test_preds.cpu().numpy())
        all_test_labels.extend(test_targets.cpu().numpy())

# Create confusion matrix
conf_matrix = confusion_matrix(all_test_labels, all_test_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
