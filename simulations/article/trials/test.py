import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF
from snntorch import utils 

class SNNModel(torch.nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()

        beta = 0.9  # neuron decay rate 
        spike_grad = surrogate.fast_sigmoid() # fast sigmoid surrogate gradient


        self.linearIn = nn.Linear(6, 10)
        self.spike1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.linearOut = nn.Linear(10, 27)
        self.spike2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

       
        self.steps = 4096;
        self.spk = torch.zeros(1)

    def forward(self, x):
        
        self.mem1 = self.spike1.init_leaky()
        self.mem2 = self.spike2.init_leaky()
        
        for step in range(self.steps):
            linear1 = self.linearIn(x[:,:,step])
            spk1, self.mem1 = self.spike1(linear1, self.mem1)
            
            linear2 = self.linearOut(spk1)
            self.spk, self.mem2 = self.spike2(linear2, self.mem2)
            
        return self.spk

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the parameters
batch_size = 4
num_channels = 6
sequence_length = 4096

# Generate fake training data
training_data = torch.randn((batch_size, num_channels, sequence_length))

# Print the shape of the generated data
print("Shape of training data:", training_data.shape)


# Assuming you have training_data and target_data
model = SNNModel()
utils.reset(model)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
num_epochs = 1

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(training_data)
    # loss = loss_fn(output, target_data)
    loss_fn.backward()
    optimizer.step()
