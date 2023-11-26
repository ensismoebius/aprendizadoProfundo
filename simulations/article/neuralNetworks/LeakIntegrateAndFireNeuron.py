"""
Summary:
This code simulates the behavior of a Leaky Integrate-and-Fire (LIF) neuron using a 
custom class. The LIFNeuron class defines the dynamics of the neuron, and the 
`simulate_lif_neuron` function conducts the simulation with specified parameters. 
The membrane potential and input current are plotted over time, demonstrating the 
response of the LIF neuron to an input current pulse. The plot includes key points 
such as the threshold, resting potential, and spike times, providing insights into 
the neuron's behavior.
"""

import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau, v_rest, v_th, v_reset, r):
        self.tau = tau          # Membrane time constant
        self.v_rest = v_rest    # Resting potential
        self.v_th = v_th        # Threshold potential
        self.v_reset = v_reset  # Reset potential
        self.r = r              # Membrane resistance
        self.v = v_rest         # Initial membrane potential
        self.spike_times = []   # To store spike times
        self.mem_potentials = []# To store membrane potentials

    def update(self, i, dt):
        # LIF neuron dynamics update using Euler's method
        dv = (-(self.v - self.v_rest) + self.r * i) / self.tau * dt
        self.v += dv

        # Check for spike condition
        if self.v >= self.v_th:
            self.spike_times.append(i)
            self.v = self.v_reset  # Reset membrane potential after spike
        
        self.mem_potentials.append(self.v)

def simulate_lif_neuron(current_input, simulation_time, dt, tau, v_rest, v_th, v_reset, r):
    neuron = LIFNeuron(tau, v_rest, v_th, v_reset, r)
    
    # Simulation loop
    for i in np.arange(0, simulation_time, dt):
        neuron.update(i, dt)
    
    return neuron

# Simulation parameters
simulation_time = 40    # in milliseconds
dt = 0.1                 # time step
tau = 10.0               # membrane time constant (ms)
v_rest = -70.0           # resting potential (mV)
v_th = -50.0             # threshold potential (mV)
v_reset = v_rest         # reset potential (mV)
r = 10.0                 # membrane resistance (Ohms)

# Input current: step function for demonstration
current_input = np.zeros(int(simulation_time/dt))
current_input[int(20/dt):int(80/dt)] = 1.0  # input current pulse

# Run simulation
neuron = simulate_lif_neuron(current_input, simulation_time, dt, tau, v_rest, v_th, v_reset, r)

# Plotting the membrane potential and input current over time
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, simulation_time, dt), neuron.mem_potentials, 'b-', label='Membrane Potential')
plt.plot([0, simulation_time], [v_th, v_th], 'k--', label='Threshold')
plt.plot([0, simulation_time], [v_rest, v_rest], 'k-', label='Resting Potential')
plt.plot([0, simulation_time], [v_reset, v_reset], 'k-.', label='Reset Potential')
plt.ylabel('Membrane Potential (mV)')
plt.title('LIF Neuron Simulation')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(neuron.spike_times, np.ones_like(neuron.spike_times), 'r|', label='Spike')
plt.xlabel('Time (ms)')
plt.legend()

plt.tight_layout()
plt.show()
