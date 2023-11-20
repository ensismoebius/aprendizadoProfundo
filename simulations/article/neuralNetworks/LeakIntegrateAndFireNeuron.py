import matplotlib.pyplot as plt
import numpy as np

def lif_neuron(membrane_time_constant, membrane_resistance, resting_potential, threshold, input_current, time_step, duration):
    num_steps = int(duration / time_step)
    time = np.arange(0, duration, time_step)

    spikes = []
    membrane_potential = resting_potential * np.ones(num_steps)

    for i in range(1, num_steps):
        dV = (-membrane_potential[i - 1] + resting_potential + membrane_resistance * input_current[i - 1]) / membrane_time_constant * time_step
        membrane_potential[i] = membrane_potential[i - 1] + dV

        if membrane_potential[i] >= threshold:
            spikes.append(i)
            membrane_potential[i] = resting_potential  # Reset membrane potential after spike

    return time, membrane_potential, spikes

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
    axs[1].set_ylabel('Spike')

    plt.tight_layout()
    plt.show()

# Parameters
membrane_time_constant = 20  # Membrane time constant (ms)
membrane_resistance = 1    # Membrane resistance
resting_potential = 0  # Resting membrane potential
threshold = 10  # Membrane potential threshold for firing
time_step = 1  # Time step (ms)
duration = 1000  # Simulation duration (ms)
input_current = np.zeros(int(duration / time_step))  # Input current (zero for simplicity)

# Set input current to non-zero values to observe different behavior
input_current[300:400] = 15.0  # Example: Injecting a current for a brief period

time, membrane_potential, spikes = lif_neuron(membrane_time_constant, membrane_resistance, resting_potential, threshold, input_current, time_step, duration)
plot_potentials_and_spikes(time, membrane_potential, spikes, threshold)
