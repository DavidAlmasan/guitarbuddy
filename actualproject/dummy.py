
import sys
import random
import math
import os     #
import pyaudio
from scipy import signal
import pygame
from socket import *
from pygame.locals import *
from random import * 
import numpy
from scipy.signal import blackmanharris, fftconvolve
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find

def loudness(chunk):
    data = numpy.array(chunk, dtype=float) / 32768.0
    ms = math.sqrt(numpy.sum(data ** 2.0) / len(data))
    if ms < 10e-8: ms = 10e-8
    return 10.0 * math.log(ms, 10.0)

class SoundRecorder:
        
    def __init__(self):
        self.RATE=48000
        self.BUFFERSIZE=4096 #1024 is a good buffer size 3072 works for Pi
        self.secToRecord=.05
        self.threadsDieNow=False
        self.newAudio=False
        
    def setup(self):
        self.buffersToRecord=int(self.RATE*self.secToRecord/self.BUFFERSIZE)
        if self.buffersToRecord==0: self.buffersToRecord=1
        self.samplesToRecord=int(self.BUFFERSIZE*self.buffersToRecord)
        self.chunksToRecord=int(self.samplesToRecord/self.BUFFERSIZE)
        self.secPerPoint=1.0/self.RATE
        self.p = pyaudio.PyAudio()
        self.inStream = self.p.open(format=pyaudio.paInt16,channels=1,rate=self.RATE,input=True,frames_per_buffer=self.BUFFERSIZE)
        self.xsBuffer=numpy.arange(self.BUFFERSIZE)*self.secPerPoint
        self.xs=numpy.arange(self.chunksToRecord*self.BUFFERSIZE)*self.secPerPoint
        self.audio=numpy.empty((self.chunksToRecord*self.BUFFERSIZE),dtype=numpy.int16)               
    
    def close(self):
        self.p.close(self.inStream)
       



    
    def getAudio(self):
        audioString=self.inStream.read(self.BUFFERSIZE)
        self.newAudio=True
        return numpy.fromstring(audioString,dtype=numpy.int16)

class AudioSequence(object):

    def main(self):
        ok = 1  # will become 0 when we want to exit application
        sequence = []
        previous = 'Start'
        stepsize = 5
        # Build frequency, noteName dictionary
        #tunerNotes = build_default_tuner_range()

        # Sort the keys and turn into a numpy array for logical indexing
        #frequencies = numpy.array(sorted(tunerNotes.keys()))

        #top_note = len(tunerNotes)-1
        bot_note = 0
        

        top_note = 24
        bot_note = 0
        
        # Misc variables for program controls
        inputnote = 1                               # the y value on the plot
        
        shownotes = True                            # note names shown or invisible
        signal_level=0                              # volume level
        fill = True                                 #
        trys = 1
        needle = False
        cls = True
        col = False
        circ = False
        line = False
        auto_scale = False
        toggle = False
        stepchange = False
        soundgate = 19                             # zero is loudest possible input level
        targetnote=0
        SR=SoundRecorder()                          # recording device (usb mic)
        
        while trys != 0:
            trys += 1
            
            while(ok):
                for event in pygame.event.get():
                    if event.type == QUIT:
                        print "DONE"
                        ok = 0
                        SR.close()
                        return
                    elif event.type == KEYDOWN:
                        if event.key == K_q:
                            print "DONE"
                            ok = 0
                            SR.close()
                            return
                
                SR.setup()
                raw_data_signal = SR.getAudio()                                         #### raw_data_signal is the input signal data 
                signal_level = round(abs(loudness(raw_data_signal)),2)     
                
                
                # try: 
                #     inputnote = round(freq_from_autocorr(raw_data_signal,SR.RATE),2)    #### find the freq from the audio sample
                    
                # except:
                #     inputnote == 0
                    
                SR.close()
                
                # if inputnote > frequencies[len(tunerNotes)-1]:                        #### not interested in notes above the notes list
                #     continue
                    
                # if inputnote < frequencies[0]:                                     #### not interested in notes below the notes list
                #     continue    
                        
                if signal_level > soundgate:                                        #### basic noise gate to stop it guessing ambient noises 
                     continue
                print "signal:" + str(raw_data_signal) + '\n'
                
                #targetnote = closest_value_index(frequencies, round(inputnote, 2))     #### find the closest note in the keyed array                

                
                if stepchange == True:                     #go to start of the loop if the step size is altered
                    stepchange = not stepchange
                    break 
                
                # display note names if selected
                # if shownotes:            
                    
                #     if previous != tunerNotes[frequencies[targetnote]]:
                #         f = open("out.txt", 'a')
                #         f.write("file " + previous + '.wav\n')
                #     previous = str(tunerNotes[frequencies[targetnote]])
                #     print(previous)

if __name__ == '__main__':
    pygame.init()
    AudioSequence().main()
