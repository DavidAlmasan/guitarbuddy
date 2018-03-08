import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *
import wave
import sys

CHUNK = 4096

wf = wave.open('Enote_5.wav', 'rb')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
fs = wf.getframerate()
X = wf.readframes(-1)
wf.close()

stream.write(X)
stream.stop_stream()
stream.close()

Y = np.fromstring(X, 'Int16')
N = len(Y)
print(N)
plt.plot(np.arange(N) / fs, Y)

# fY = fft(Y[CHUNK:2*CHUNK])
a = CHUNK
fY = fft(Y[a:a+CHUNK])
plt.figure()
plt.plot(np.arange(CHUNK) * fs / CHUNK, np.abs(fY))
plt.show()

p.terminate()