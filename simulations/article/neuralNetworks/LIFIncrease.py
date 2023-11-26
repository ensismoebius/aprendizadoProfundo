"""
Summary:
This code simulates the membrane potential increase in a Leaky Integrate-and-Fire (LIF) 
neuron model. The `lifIncrease` function iteratively computes the membrane potential 
over a specified number of time steps. The results are plotted to visualize the change 
in membrane potential over time. The code provides insights into the behavior of a 
simple LIF neuron in response to an input current, demonstrating the characteristic 
increase in membrane potential.
"""

import math
import matplotlib.pyplot as plt

def lifIncrease(V_mem, dt=1, I_in=1, R=5, C=1):
	tau = R*C
	V_mem = V_mem + (dt/tau)*(-V_mem + I_in*R)
	return V_mem

num_steps = 40
V_mem = 0.9
V_trace = []  # keeps a record of U for plotting

for step in range(num_steps):
  V_trace.append(V_mem)
  V_mem = lifIncrease(V_mem)  # solve next step of U

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(V_trace, label='Membrane Potential (mV)')
plt.title('LIF potential increase')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.tight_layout()
plt.show()
