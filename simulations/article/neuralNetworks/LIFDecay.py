"""
Summary:
This code simulates the membrane potential decay in a Leaky Integrate-and-Fire (LIF) 
neuron model. The `lifDecay` function iteratively computes the membrane potential over 
a specified number of time steps with a decaying input. The results are plotted to 
visualize the decrease in membrane potential over time. The code provides insights 
into the behavior of a simple LIF neuron when subjected to a zero input current.
"""

import math
import matplotlib.pyplot as plt

def lifDecay(V_mem, dt=1, I_in=0, R=5, C=1):
	tau = R*C
	V_mem = V_mem + (dt/tau)*(-V_mem + I_in*R)
	return V_mem

num_steps = 40
V_mem = 0.9
V_trace = []  # keeps a record of U for plotting

for step in range(num_steps):
  V_trace.append(V_mem)
  V_mem = lifDecay(V_mem)  # solve next step of U

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(V_trace, label='Membrane Potential (mV)')
plt.title('LIF potential decay')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.tight_layout()
plt.show()
