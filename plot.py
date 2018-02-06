import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import urllib

# Read frequency and data array for sound track
local_filename = 'output.wav'
fs, x = scipy.io.wavfile.read(local_filename) 

# If we have a stero track (left and right channels), take just the first channel
if len(x.shape) > 1:
    x = x[:, 0]

# Time points (0 to T, with T*fs points)
t = np.linspace(0, len(x)/fs, len(x), endpoint=False)

# Plot signal
plt.plot(t, x)
cut = [(t0, x0) for (t0, x0) in zip(t, x) if (0.49 <= t0 and t0 <= 1.32)]
t = [c[0] for c in cut]
x = [c[1] for c in cut]
plt.xlabel('time (seconds)')
plt.ylabel('signal')
plt.show()

# Perform discrete Fourier transform (real signal)
xf = np.fft.rfft(x)

# Create frequency axis for plotting
freq = np.linspace(0.0, fs/2, len(xf))

plt.semilogy(freq, np.abs(xf))
plt.xlabel('frequency (Hz)')
plt.ylabel('$\hat{x}$');
plt.show()