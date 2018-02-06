"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys

CHUNK = 1024

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

print('starting')
wf = wave.open(sys.argv[1], 'rb')
print('file done reading')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

print('string created')

data = wf.readframes(CHUNK)

print('step 1')

while data != '':
    stream.write(data)
    data = wf.readframes(CHUNK)

print('stopping')

stream.stop_stream()
stream.close()

p.terminate()
