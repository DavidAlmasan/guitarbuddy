import numpy as np
import matplotlib.pyplot as plt
import wave
import pyaudio
import sys

pa = pyaudio.PyAudio()

wf = wave.open('Enote_0.wav', 'rb')

def create(L, name):
    wf = wave.open(name, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(L))
    wf.close()

# def pwr(Y, i, pwrRange = 4096, thr = 500):
#     A = Y[i:i+pwrRange]
#     P = sum([a**2 for a in A]) / pwrRange
#     P = np.sqrt(P)
#     return P

paramnames = ['nchannels', 'sampwidth', 'framerate', 'nframes', 'comptype', 'compname']
p = wf.getparams()
for i,j in zip(paramnames, p):
    print(str(i) + ': ' + str(j))
Y = wf.readframes(-1)
fs = wf.getframerate()
Y = np.fromstring(Y, 'Int16')
# plt.plot(np.arange(len(Y)) / fs, Y)
plt.plot(Y)

# Detect silences and split the file
pwrRange = 4096
thr = 500
P = np.zeros(len(Y) - pwrRange + 1)
P[0] = sum([a**2 for a in Y[:pwrRange]]) / pwrRange
for i in range(1, len(Y) - pwrRange + 1):
    P[i] = P[i-1] + (Y[i+pwrRange-1]**2 - Y[i-1]**2) / pwrRange
plt.grid()
# P = [pwr(Y, i, pwrRange) for i in range(len(Y) - pwrRange + 1)]
P = np.sqrt(P)
P = [0 if p < thr else p for p in P]
i =0
index = 0
while i < len(P)-1:
    if P[i] == 0 and P[i+1]!=0:
        start = i+1
        print(start)
        j = start + 1
        while j<len(P)-1:
            if P[j] != 0 and P[j+1] ==0:
                stop = j+1
                create(Y[start:stop], 'Enote_'+str(index)+'.wav')
                index += 1
                break
            j += 1
    i+=1
plt.figure()
plt.plot(P)
plt.show()
# create(Y, 'samptest.wav')
