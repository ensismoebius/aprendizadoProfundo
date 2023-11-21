import matplotlib.pyplot as plt
import numpy as np

def lif_neuron(
    tau, # membrane time constant 
    R, # membrane resistance
    V_rest, # resting potential 
    V_thresh, # potential threshold
    I_in, # input current
    dt, # time step 
    duration
    ):
    
    # Auxiliary variables
    spikes = []
    num_steps = int(duration / dt)
    time = np.arange(0, duration, dt)
    
    # membrane potentials
    V_mem = V_rest * np.ones(num_steps)

    # Here the LIF model actually begins
    for i in range(1, num_steps):
        
        
        # def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
        #     tau = R*C
        #     U = U + (time_step/tau)*(-U + I*R)
        #     return U
        
        dV = (-V_mem[i - 1] + V_rest + R * I_in[i - 1]) / tau * dt
        V_mem[i] = V_mem[i - 1] + dV

        if V_mem[i] >= V_thresh:
            spikes.append(i)
            V_mem[i] = V_rest  # Reset membrane potential after spike

    return time, V_mem, spikes

def plot_potentials_and_spikes(time, membrane_potential, spikes, threshold):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    axs[0].plot(time, membrane_potential, label='Membrane Potential')
    axs[0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    axs[0].set_title('LIF Neuron Membrane Potential')
    axs[0].set_ylabel('Membrane Potential')
    axs[0].legend()

    axs[1].eventplot(spikes, color='black', linewidths=2)
    axs[1].set_title('Spikes')
    axs[1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()

# Parameters
membrane_time_constant = 80  # Membrane time constant in ms (τ tau)
membrane_resistance = 1.0    # Membrane resistance
resting_potential = -2.0  # Resting membrane potential
threshold = 10.0  # Membrane potential threshold for firing
time_step = 1.0  # Time step (ms)
duration = 1000.0  # Simulation duration (ms)
input_current = np.zeros(int(duration / time_step))  # Input current (zero for simplicity)

# Set input current to non-zero values to observe different behavior
input_current[300:800] = 15.0  # Example: Injecting a current for a brief period

time, membrane_potential, spikes = lif_neuron(membrane_time_constant, membrane_resistance, resting_potential, threshold, input_current, time_step, duration)
plot_potentials_and_spikes(time, membrane_potential, spikes, threshold)