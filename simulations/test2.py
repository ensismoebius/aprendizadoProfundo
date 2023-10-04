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

        # Time step for decaying (still do not known what is this)
        self.timeStep = 0.1
        
        # The by which the manbrane voltage decays each time step
        self.alpha = np.exp(-self.timeStep/self.tau)
        
        

    def fire_spike(self):
        self.voltage = 0
        return 1

    def add_synaptic_weight(self, weigth):
        # Membrane voltage integration
        self.voltage += weigth

    def iterate(self):
        # Membrane voltage leak
        self.voltage = max(self.voltage * self.alpha, 0)


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
        self.fig, self.ax = plt.subplots()
        self.x = np.linspace(0, 1000, 1000)
        self.y = np.zeros(1000)

        self.line, = self.ax.plot(self.x, self.y)

        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Membrane voltage')
        self.ax.set_title('Membrane voltage x time')

        self.canvas = FigureCanvas(self.fig)
        vbox.pack_start(self.canvas, expand=True, fill=True, padding=5)

        self.add(vbox)

        # Initialize the animation
        self.animation_id = GLib.idle_add(self.update_plot)

    def on_spike_button_clicked(self, button):
        self.neuron.add_synaptic_weight(1.0)

    def update_plot(self):
        self.x[:-1] = self.x[1:]
        self.x[-1] = self.x[-1] + 1

        self.y[:-1] = self.y[1:]
        self.y[-1] = self.neuron.voltage

        self.line.set_data(self.x, self.y)  # Update the plot data

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()  # Draw the updated plot

        self.neuron.iterate()
        return True


if __name__ == "__main__":
    win = SineWaveApp()
    win.show_all()
    Gtk.main()
