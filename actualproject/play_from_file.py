import pyaudio
import wave
import sys
import os


class AudioFile:
    chunk = 1024 * 5

    def __init__(self, file):
        """ Init audio stream """ 
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """ 
        self.stream.close()
        self.p.terminate()
tosound = { "A2":"A2.wav", 
            "B2":"B2.wav", 
            "C2":"C2.wav",}
# Usage example for pyaudio
song = []
file = open("out.txt", 'r')
for line in file:
    if line != 'Start\n':
        song.append(line)
        song[len(song)-1] = song[len(song)-1][:2]
# for note in song:
#     a = AudioFile(tosound[note])
#     a.play()
#     a.close()
os.system('ffmpeg -f concat -i out.txt -c copy output.wav')
a = AudioFile("output.wav")
a.play()
a.close()