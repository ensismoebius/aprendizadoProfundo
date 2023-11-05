import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def get_tensor_size_after_conv(input_size, convolution_kernel_size):
    return 1 +(input_size - convolution_kernel_size)

def get_tensor_size_after_maxpool(input_size, poll_kernel_size):
    return math.floor(input_size / poll_kernel_size)


# Define a simple neural network architecture with dynamic size calculation
class EEGNet(nn.Module):
    def __init__(self, input_channels, input_timepoints):
        super(EEGNet, self).__init__()
        
        # Calculate the size of the features after the 1st conv and pool layers
        self.feature_size = get_tensor_size_after_conv(input_timepoints,5)
        self.feature_size = get_tensor_size_after_maxpool(self.feature_size,2)
        # Calculate the size of the features after the 2nd conv and pool layers
        self.feature_size = get_tensor_size_after_conv(self.feature_size,5)
        self.feature_size = get_tensor_size_after_maxpool(self.feature_size,2)
        # Calculate the size of the features after the flatten
        self.feature_size = self.feature_size * 32
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 27)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.fl1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Create the neural network
input_channels = 4  # number of EEG channels
input_timepoints = 250  # number of time points in each EEG sample
net = EEGNet(input_channels=input_channels, input_timepoints=input_timepoints)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Prepare your dataset
# Example (dummy data and labels):
X_dummy = torch.randn(27, 4, 250)  # 100 samples of 4-channel EEG data
y_dummy = torch.linspace(0, 26, steps=27)  # 26 character labels

# Create a DataLoader instance
dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

# Plot the EEG data (4 channels) for a single sample
plt.figure(figsize=(15, 5))
for i in range(input_channels):
    plt.plot(X_dummy[0, i, :], label=f'Channel {i+1}')
plt.legend()
plt.title('Sample EEG Data (4 channels)')
plt.xlabel('Time Points')
plt.ylabel('Amplitude')
plt.show()
