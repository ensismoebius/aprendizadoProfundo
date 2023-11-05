import math
import torch
import torch.nn as nn
from torch import nn, save, load
from torch.utils.data import DataLoader, TensorDataset
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
                    print('[%d, %5d] loss: %.3f' %
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
input_channels = 4  # number of EEG channels
input_timepoints = 250  # number of time points in each EEG sample
net = EEGNet(input_channels=input_channels, input_timepoints=input_timepoints)

# Prepare your dataset
# Example (dummy data and labels):
X_dummy = torch.randn(26, 4, 250)  # 100 samples of 4-channel EEG data
y_dummy = torch.linspace(0, 25, steps=26)  # 26 character labels

# Create a DataLoader instance
dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

net.start_train(dataloader, 200, 0.0001)
net.save_model('test.pth')
net.load_model('test.pth')

# This selects the first sample and retains all channels, adding an extra dimension for the batch.
print(torch.argmax(net(X_dummy[10].unsqueeze(0)), dim=1))
