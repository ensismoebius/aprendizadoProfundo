\begin{lstlisting}[language=Python]
import numpy as np
import matplotlib.pyplot as plt

class LiquidTimeConstantNeuron:
    def __init__(self):
        self.time_constant = 1.0

    def adjust_time_constant(self, input_data):
        # Adjust the time constant based on input data
        self.time_constant = 1.0 + input_data

    def update_state(self, state, time_step):
        # Update the state based on the time constant and time step
        return state - (state / self.time_constant) + np.sin(time_step)

# Create a Liquid Time-Constant Neuron
ltc_neuron = LiquidTimeConstantNeuron()

# Simulate over time
duration = 10.0
time = np.arange(0, duration, 0.1)
states = [0.0]

for t in time[1:]:
    input_data = np.sin(t)  # Simulated input data (can vary over time)
    ltc_neuron.adjust_time_constant(input_data)
    new_state = ltc_neuron.update_state(states[-1], t)
    states.append(new_state)

# Plot the results
plt.plot(time, states)
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Neuron with Adjustable Time Constant")
plt.show()
\end{lstlisting}
\begin{lstlisting}[language=Python]
import torch
import torch.nn as nn

class ESN(nn.Module):
def __init__(self, input_size, reservoir_size, output_size):
super(ESN, self).__init__()
self.reservoir_size = reservoir_size
self.W_in = nn.Linear(input_size, reservoir_size)
self.W_res = nn.Linear(reservoir_size, reservoir_size)
self.W_out = nn.Linear(reservoir_size, output_size)

def forward(self, input):
reservoir = torch.zeros((input.size(0), self.reservoir_size))
for i in range(input.size(1)):
input_t = input[:, i, :]
reservoir = torch.tanh(self.W_in(input_t) + self.W_res(reservoir))
output = self.W_out(reservoir)
return output

# Example usage
input_size = 10
reservoir_size = 100
output_size = 1

model = ESN(input_size, reservoir_size, output_size)
\end{lstlisting}