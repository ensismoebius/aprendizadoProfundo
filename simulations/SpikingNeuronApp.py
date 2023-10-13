from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from gi.repository import Gtk, GLib
import numpy as np
import matplotlib.pyplot as plt
import gi
import pyaudio
import threading

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

        # The threshold above which the neuron fires
        self.threshold = threshold

        # Time step for decaying (i still don´t known what this really is)
        # Bigger the number faster the decay
        self.timeStep = 0.5

        # The rate by which the membrane voltage decays each time step
        self.alpha = np.exp(-self.timeStep/self.tau)

    def set_tau(self, tau):
        self.tau = tau
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


class AudioReceiver:
    def __init__(self):
        self.CHUNK = 64
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.audio = pyaudio.PyAudio()

    def beginStream(self, callback):
        self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                                      rate=self.RATE, input=True,
                                      frames_per_buffer=self.CHUNK)

        def capture_and_process_audio():
            while True:
                data = self.stream.read(self.CHUNK)
                data = np.frombuffer(data, dtype=np.int16)
                # Normalize the audio data between 0 and 1
                normalized_data = np.abs(data) / np.iinfo(np.int16).max
                callback(normalized_data)

        # Create and start a new thread for audio capture and processing
        audio_thread = threading.Thread(target=capture_and_process_audio)
        audio_thread.daemon = True  # Allow the thread to be terminated when the program exits
        audio_thread.start()



class SpikingNeuronApp(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Spike neuron simulation")
        self.connect("destroy", Gtk.main_quit)

        self.neuron = Neuron()

        self.audioReceiver = AudioReceiver()

        vbox = Gtk.VBox()

        # Create sliders for adjusting tau and threshold
        self.tau_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=Gtk.Adjustment(
                value=10,
                lower=1,
                upper=100,
                step_increment=1,
                page_increment=10,
                page_size=0)
        )
        self.tau_scale.set_digits(0)
        self.tau_scale.set_valign(Gtk.Align.START)
        self.tau_scale.connect("value-changed", self.on_tau_changed)
        self.tau_scale_label = Gtk.Label(label="Tau")
        vbox.pack_start(self.tau_scale_label, expand=False,
                        fill=False, padding=5)
        vbox.pack_start(self.tau_scale, expand=False, fill=True, padding=5)

        self.threshold_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=Gtk.Adjustment(
                value=1,
                lower=0,
                upper=10,
                step_increment=0.1,
                page_increment=1,
                page_size=0)
        )
        self.threshold_scale.set_digits(1)
        self.threshold_scale.set_valign(Gtk.Align.START)
        self.threshold_scale.connect(
            "value-changed", self.on_threshold_changed)
        self.threshold_scale_label = Gtk.Label(label="Threshold")
        vbox.pack_start(self.threshold_scale_label,
                        expand=False, fill=False, padding=5)
        vbox.pack_start(self.threshold_scale,
                        expand=False, fill=True, padding=5)

        spike_label = Gtk.Label(label="Spike Amplitude:")
        vbox.pack_start(spike_label, expand=False, fill=False, padding=5)

        self.spike_button = Gtk.Button(label="Add Spike")
        self.spike_button.connect("clicked", self.on_spike_button_clicked)
        vbox.pack_start(self.spike_button, expand=False, fill=False, padding=5)
        
        self.mic_button = Gtk.Button(label="Microphone capture")
        self.mic_button.connect("clicked", self.on_mic_button_clicked)
        vbox.pack_start(self.mic_button, expand=False, fill=False, padding=5)

        # Create a Matplotlib figure and add it to the GTK window
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]})

        # Increase the vertical spacing between subplots using subplots_adjust
        plt.subplots_adjust(hspace=0.4)

        self.time = np.linspace(0, 1000, 1000)
        self.voltages = np.zeros(1000)
        self.spike_activity = np.zeros(1000)

        self.threshold_line = None  # Store the threshold line object
        self.threshold_label = None  # Store the threshold label object

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


    def on_mic_data_received(self, data):
        # Calculate some value from the audio data that you want to use as input
        audio_input_value = np.mean(data)  # You can adjust this calculation
    
        # Add the audio input value as a synaptic weight to the neuron
        self.neuron.add_synaptic_weight(audio_input_value)

    def on_mic_button_clicked(self, button):
        self.audioReceiver.beginStream(self.on_mic_data_received)

    def on_spike_button_clicked(self, button):
        self.neuron.add_synaptic_weight(1.0)

    def on_tau_changed(self, scale):
        self.neuron.set_tau(scale.get_value())

    def on_threshold_changed(self, scale):
        self.neuron.threshold = scale.get_value()

    def update_plot(self):
        self.time[:-1] = self.time[1:]
        self.time[-1] = self.time[-1] + 1

        self.voltages[:-1] = self.voltages[1:]
        self.voltages[-1] = self.neuron.voltage

        self.spike_activity[:-1] = self.spike_activity[1:]
        self.spike_activity[-1] = self.neuron.iterate()

        if self.threshold_line is None:
            # Add the threshold line if it doesn't exist
            self.threshold_line = self.ax1.axhline(
                y=self.neuron.threshold, color='r', linestyle='--', label='Threshold')
            self.threshold_label = self.ax1.text(
                0, self.neuron.threshold, 'Threshold', color='r')

        # Update the position of the threshold label based on the current x-axis limits
        thresholdPosition, _ = self.ax1.get_xlim()
        self.threshold_label.set_position(
            (thresholdPosition, self.neuron.threshold + 0.01))

        # Update the threshold line position
        self.threshold_line.set_ydata(
            [self.neuron.threshold, self.neuron.threshold])

        # Update the voltage plot
        self.line1.set_data(self.time, self.voltages)
        # Update the spike activity plot
        self.line2.set_data(self.time, self.spike_activity)

        self.ax1.relim()
        self.ax1.autoscale_view()

        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw_idle()  # Draw the updated plots

        return True


if __name__ == "__main__":
    win = SpikingNeuronApp()
    win.show_all()
    Gtk.main()
