import pyaudio
import matplotlib.pyplot as plt
import numpy as np

# Parameters for audio input
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open a streaming stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Create a window for plotting
plt.ion()
fig, ax = plt.subplots()
x = np.arange(0, CHUNK)
line, = ax.plot(x, np.random.rand(CHUNK))

# Set plot labels and title
ax.set_xlabel('Sample Number')
ax.set_ylabel('Amplitude')
ax.set_title('Microphone Audio')

# Set y-axis limits to fit between 0 and 1
ax.set_ylim(0, 1)

try:
    while True:
        # Read audio data from the microphone
        data = stream.read(CHUNK)
        data = np.frombuffer(data, dtype=np.int16)

        # Normalize the audio data to the range [0, 1]
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Update the plot with the new data
        line.set_ydata(normalized_data)
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    pass

# Close the audio stream and terminate PyAudio
stream.stop_stream()
stream.close()
audio.terminate()

# Close the plot window
plt.close()
