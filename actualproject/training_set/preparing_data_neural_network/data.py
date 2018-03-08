import numpy as np
import matplotlib.pyplot as plt
import wave
import pyaudio
import sys
import csv
import csv

pa = pyaudio.PyAudio()

f = open("data_set.txt", 'w')
csvfile = open("dataset.csv", 'wb')
spamwriter = csv.writer(csvfile, delimiter=',')

wf = wave.open('E_0.wav', 'rb')

CHUNK = 1023 + 1 #+1 to account for losing one number at the end which becomes the label
paramnames = ['nchannels', 'sampwidth', 'framerate', 'nframes', 'comptype', 'compname']
p = wf.getparams()
for i,j in zip(paramnames, p):
    print(str(i) + ': ' + str(j))
Y = wf.readframes(-1)
fs = wf.getframerate()
Y = np.fromstring(Y, 'Int16')
index = 0
# while index+CHUNK-1<len(Y):
#     f = open("data_set.txt", 'a')
#     f.write(str(Y[index:index+CHUNK])+'\n')
#     index += CHUNK/8

index = 0
while index+ CHUNK < len(Y):
    row = Y[index:index+CHUNK]
    row[CHUNK-1] = 0
    spamwriter.writerow([a for a in row])   #0 indicative of E note, 1 idicative of G note
    index +=CHUNK/8



