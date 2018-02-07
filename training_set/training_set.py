import sys
import os

EXT = '.txt'
INPUT = 'output.wav'

def next_chord():
    return 'Eminor'

def next_file():
    pass

def read_wave(filename):
    pass

def split_wave(wavefile):
    return [1, 1, 1, 1, 1, 1]

D = os.listdir()

if len(sys.argv) == 1:
    chord = next_chord()
elif len(sys.argv) == 2:
    chord = sys.argv[1]
else:
    chord = None

if chord not in D and chord != None:
    os.makedirs(chord)
else:
    print('chord exists')

chord_dir = os.listdir(str(chord))
L = len(chord_dir)
i = 0
wavs = split_wave(read_wave(INPUT))
for w in wavs:
    while (str(i) + EXT) in chord_dir: i += 1
    f = open(chord + '/' + (str(i) + EXT), 'w')
    f.write('cont\n')
    f.close()
    i += 1
