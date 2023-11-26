import math
import matplotlib.pyplot as plt
import numpy as np

def lif(V_mem, dt=1, I_in=1, R=5, C=1, V_thresh = 2, reset_zero = True):
	tau = R*C
	V_mem = V_mem + (dt/tau)*(-V_mem + I_in*R)
 
	if V_mem > V_thresh:
		if reset_zero:
			V_mem = 0
		else:
			V_mem = V_mem - V_thresh

 
	return V_mem

input = np.concatenate((np.zeros(50), np.ones(20)*(.5), np.zeros(50)), 0)
num_steps = len(input)
V_mem = 0
V_trace = []

for step in range(num_steps):
  V_trace.append(V_mem)
  V_mem = lif(V_mem, I_in=input[step])

# Plot results
plt.figure(figsize=(10, 4))

plt.subplot(2, 1, 1)
plt.plot(V_trace, label='Membrane Potential (mV)')
plt.title('LIF potential')
plt.ylabel('Voltage (mV)')
plt.xlabel('Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.title('LIF input')
plt.plot(input, label='Input current (mA)')
plt.ylabel('Current (mA)')
plt.xlabel('Time')
plt.legend()

plt.tight_layout()
plt.show()