from numpy import *
from numpy.fft import *
import matplotlib.pyplot as plt
from sys import *
import pyaudio
import datetime
import wave

CHUNK = 1024
fs = 44100
N = 2**17
T0 = N / fs
print(T0)
T = linspace(0, T0, N, endpoint = False)

X = sin(2*pi*440*T)
plt.plot(arange(N) / T0, abs(fft(X)))
# plt.show()


# instantiate PyAudio (1)
p = pyaudio.PyAudio()
X = float32(X)

# open stream (2)
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# # read data
# data = wf.readframes(CHUNK)
t0 = datetime.datetime.now()
stream.write(X)
print((datetime.datetime.now() - t0))
# # play stream (3)
# while len(data) > 0:
#     stream.write(data)
#     data = wf.readframes(CHUNK)

wf = wave.open('test.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
wf.setframerate(fs)
wf.writeframes(b''.join(X))
wf.close()

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()