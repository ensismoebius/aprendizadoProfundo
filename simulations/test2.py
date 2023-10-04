import numpy as np
import matplotlib.pyplot as plt
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

class Neuron:
    def __init__(self):
        self.voltage = 1

    def add_spike(self, spike):
        self.voltage += spike

    def iterate(self):
        self.voltage = max(self.voltage - 0.01, 0)

class SineWaveApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Spike neuron simulation")
        self.connect("destroy", Gtk.main_quit)

        self.neuron = Neuron()

        vbox = Gtk.VBox()

        spike_label = Gtk.Label(label="Spike Amplitude:")
        vbox.pack_start(spike_label, expand=False, fill=False, padding=5)

        self.spike_button = Gtk.Button(label="Add Spike")
        self.spike_button.connect("clicked", self.on_spike_button_clicked)
        vbox.pack_start(self.spike_button, expand=False, fill=False, padding=5)


        # Create a Matplotlib figure and add it to the GTK window
        self.fig, self.ax = plt.subplots()
        self.x = np.linspace(0, 1000, 1000)
        self.y = np.zeros(1000)
        
        self.line, = self.ax.plot(self.x, self.y)
        
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Voltage')
        self.ax.set_title('Spike neuron simulation')

        self.canvas = FigureCanvas(self.fig)
        vbox.pack_start(self.canvas, expand=True, fill=True, padding=5)

        self.add(vbox)

        # Initialize the animation
        self.animation_id = GLib.idle_add(self.update_plot)

    def on_spike_button_clicked(self, button):
        self.neuron.add_spike(1.0)

    def update_plot(self):
        self.x[:-1] = self.x[1:]
        self.x[-1] = self.x[-1] + 1

        self.y[:-1] = self.y[1:]
        self.y[-1] = self.neuron.voltage

        self.neuron.iterate()

        self.line.set_data(self.x, self.y)  # Update the plot data

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()  # Draw the updated plot

        return True

if __name__ == "__main__":
    win = SineWaveApp()
    win.show_all()
    Gtk.main()
