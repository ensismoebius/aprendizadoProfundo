import numpy as np
import matplotlib.pyplot as plt
import time
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

# Function to update the plot
def update_plot(i):
    spike_amplitude = spike_amplitude_adjustment.get_value()
    plt.clf()
    t = np.linspace(0, 2 * np.pi, 1000)
    y = np.sin(t + 0.1 * i) + spike_amplitude * np.sin(5 * t)  # Add a temporary spike
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Infinite Animated Sine Wave with Spike')
    plt.pause(0.01)  # Adjust the pause duration to control animation speed
    plt.draw()

# Create a GTK window
class SineWaveApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Sine Wave Animation")
        self.connect("destroy", Gtk.main_quit)

        # Create a VBox container
        vbox = Gtk.VBox()

        # Create a label for the spike amplitude
        spike_label = Gtk.Label(label="Spike Amplitude:")
        vbox.pack_start(spike_label, expand=False, fill=False, padding=5)

        # Create a adjustment for the spike amplitude
        self.spike_amplitude_adjustment = Gtk.Adjustment(value=0.0, lower=0.0, upper=2.0, step_increment=0.1)
        self.spike_amplitude_adjustment.connect("value-changed", self.on_spike_amplitude_changed)

        # Create a scale widget for the spike amplitude
        self.spike_amplitude_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.spike_amplitude_adjustment)
        vbox.pack_start(self.spike_amplitude_scale, expand=False, fill=False, padding=5)

        self.add(vbox)

        # Create a figure
        self.fig = plt.figure()

        # Infinite animation loop
        self.i = 0
        GLib.idle_add(self.update_plot)

    def on_spike_amplitude_changed(self, adjustment):
        self.update_plot()

    def update_plot(self):
        spike_amplitude = self.spike_amplitude_adjustment.get_value()
        plt.clf()
        t = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(t + 0.1 * self.i) + spike_amplitude * np.sin(5 * t)  # Add a temporary spike
        plt.plot(t, y)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Infinite Animated Sine Wave with Spike')
        plt.pause(0.01)  # Adjust the pause duration to control animation speed
        plt.draw()
        self.i += 1
        return True

# Create and run the GTK application
win = SineWaveApp()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()

