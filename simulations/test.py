import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import ttk

# Function to update the plot
def update_plot(i):
    spike_amplitude = spike_amplitude_slider.get()
    plt.clf()
    t = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(t + 0.1 * i) + spike_amplitude * np.sin(5 * t)  # Add a temporary spike
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Infinite Animated Sine Wave with Spike')
    plt.pause(0.01)  # Adjust the pause duration to control animation speed

# Create a tkinter window
root = tk.Tk()
root.title('Sine Wave Animation')

# Create a frame for the slider
frame = ttk.Frame(root)
frame.pack(padx=20, pady=20)

# Create a Label for the spike amplitude
spike_label = ttk.Label(frame, text='Spike Amplitude:')
spike_label.pack()

# Create a slider widget for the spike amplitude
spike_amplitude_slider = ttk.Scale(frame, from_=0, to=2, length=200, orient='horizontal')
spike_amplitude_slider.set(0.0)
spike_amplitude_slider.pack()

# Create a figure
fig = plt.figure()

# Infinite animation loop
i = 0
while True:
    update_plot(i)
    i += 1

# Start the tkinter main loop
root.mainloop()

