"""
Summary:
This code implements a Spiking Neural Network (SNN) using the snntorch library
for EEG classification. The SNNModel is defined with leaky integrate-and-fire 
neurons and average pooling. The script includes data loading, model training, 
and evaluation on a test set. Training details such as optimization, loss computation, 
and performance metrics are provided. Visualizations, including accuracy and loss 
plots, as well as a confusion matrix, are generated and saved. 
The trained SNN model is then saved for future use.
"""
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
    
# Set random seed for reproducibility
torch.manual_seed(42)

# Load and preprocess data
data_path = '/content/drive/MyDrive/databases/S01_EEG.mat'
eegDataset = EEGDataset(data_path)

# Create model
insist = 4
neuronDecayRate = 0.9
model = SNNModel(
    inputLength=eegDataset.channelsLength, 
    insist=insist, 
    neuronDecayRate=neuronDecayRate
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.00001, 0.999))
loss_fn = nn.CrossEntropyLoss()

# Ensure that the network runs with floats
model.float()

# Get the indices for the entire dataset
dataset_size = len(eegDataset)
indices = list(range(dataset_size))

# Split the indices into training and validation sets
train_indices, test_indices = train_test_split(indices, test_size=0.4, random_state=42, shuffle=True)

# Create SubsetRandomSamplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders using the SubsetRandomSamplers
train_loader = DataLoader(eegDataset, batch_size=10, sampler=train_sampler)
test_loader = DataLoader(eegDataset, batch_size=10, sampler=test_sampler)

# Loss and accuracy history
loss_hist = []  # record loss over iterations
accuracy_hist = []  # record accuracy over iterations

# Keeps the minimum loss
min_loss = 1000000

# Keeps the maximum loss
max_loss = 0

# Training loop
num_epochs = 3000
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
        spk_rec, _ = model(data)

        loss_val = torch.zeros(1, dtype=torch.float, device=device)
        loss_val += loss_fn(spk_rec.sum(2).sum(0).squeeze(), targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Compute accuracy and update confusion matrix
        _, predicted = torch.max(spk_rec.sum(2),0)
        correct_preds += (predicted == targets).sum().item()
        total_samples += targets.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    # Calculate accuracy and loss
    accuracy = correct_preds / total_samples
    avg_loss = loss_val.item() / len(train_loader)

    # Update history
    accuracy_hist.append(accuracy)
    loss_hist.append(avg_loss)

    # Print and store results every 10 epochs
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Train Accuracy: {accuracy:.2%}, Train Loss: {avg_loss:.4f}")

# Save model
model.save_model("snn.pth")

# Print max accuracy and min loss
print(f"Maximum Accuracy: {max(accuracy_hist):.2%}")
print(f"Minimum Loss: {min(loss_hist):.4f}")

# Plot average accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(accuracy_hist, label='Accuracy')
plt.yscale('log')
plt.title('Average Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_hist, label='Loss')
plt.yscale('log')
plt.title('Average Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("SnnAccuLoss.pdf")
plt.show()

# Evaluate on the test set
model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for test_data, test_targets in test_loader:
        test_data, test_targets = test_data.to(device), test_targets.to(device)
        test_outputs, _ = model(data)
        _, test_preds = torch.max(spk_rec.sum(2),0)
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
plt.savefig("SnnConfusinMatrix.pdf")
plt.show()
