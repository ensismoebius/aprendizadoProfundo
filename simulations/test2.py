from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from gi.repository import Gtk, GLib
import numpy as np
import matplotlib.pyplot as plt
import gi

gi.require_version("Gtk", "3.0")


class Neuron:
    """
    Leak integrate and fire neuron - LIF

    """

    def __init__(self, tau=10, threshold=1):

        # Initial membrane voltage
        self.voltage = 0

        # The smaller tau is the faster the voltage decays
        # When tau is large the neuron acts as an intergrator summing its inputs
        # and firing when a certain threshold is reached.
        # When tau is small the neuron acts as a coincidence detector, firing a
        # spike only when two or more input arrive simultaneosly.
        self.tau = tau

        # The threshold above what the neuron spikes
        self.threshold = threshold

        # Time step for decaying (still do not known what is this)
        self.timeStep = 0.1

        # The by which the membrane voltage decays each time step
        self.alpha = np.exp(-self.timeStep/self.tau)

    def fire_spike(self):
        if self.voltage > self.threshold:
            self.voltage = 0
            return 1
        return 0

    def add_synaptic_weight(self, weigth):
        # Membrane voltage integration
        self.voltage += weigth

    def iterate(self):
        # Membrane voltage leak
        self.voltage = max(self.voltage * self.alpha, 0)
        return self.fire_spike()


class SineWaveApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Spike neuron simulation")
        self.connect("destroy", Gtk.main_quit)

        self.neuron = Neuron(1)

        vbox = Gtk.VBox()

        spike_label = Gtk.Label(label="Spike Amplitude:")
        vbox.pack_start(spike_label, expand=False, fill=False, padding=5)

        self.spike_button = Gtk.Button(label="Add Spike")
        self.spike_button.connect("clicked", self.on_spike_button_clicked)
        vbox.pack_start(self.spike_button, expand=False, fill=False, padding=5)

        # Create a Matplotlib figure and add it to the GTK window
        # self.fig, self.ax = plt.subplots()
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]})

        self.time = np.linspace(0, 1000, 1000)
        self.voltages = np.zeros(1000)
        self.spike_activity = np.zeros(1000)

        # self.line, = self.ax.plot(self.time, self.voltages)
        self.line1, = self.ax1.plot(self.time, self.voltages)
        self.line2, = self.ax2.plot(self.time, self.spike_activity)

        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Membrane voltage')
        self.ax1.set_title('Membrane voltage x time')

        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Spike Activity')
        self.ax2.set_title('Spike Activity x time')

        self.canvas = FigureCanvas(self.fig)
        vbox.pack_start(self.canvas, expand=True, fill=True, padding=5)

        self.add(vbox)

        # Initialize the animation
        self.animation_id = GLib.idle_add(self.update_plot)

    def on_spike_button_clicked(self, button):
        self.neuron.add_synaptic_weight(1.0)

    def update_plot(self):
        self.time[:-1] = self.time[1:]
        self.time[-1] = self.time[-1] + 1

        self.voltages[:-1] = self.voltages[1:]
        self.voltages[-1] = self.neuron.voltage

        self.spike_activity[:-1] = self.spike_activity[1:]
        self.spike_activity[-1] = self.neuron.iterate()

        self.line1.set_data(self.time, self.voltages)  # Update the voltage plot
        # Update the spike activity plot
        self.line2.set_data(self.time, self.spike_activity)

        self.ax1.relim()
        self.ax1.autoscale_view()

        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw_idle()  # Draw the updated plots

        return True


if __name__ == "__main__":
    win = SineWaveApp()
    win.show_all()
    Gtk.main()
