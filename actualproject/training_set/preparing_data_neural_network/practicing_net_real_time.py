import pandas as ps 
import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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


class SoundRecorder:
        
    def __init__(self):
        self.RATE=44200
        self.BUFFERSIZE=1023 #1024 is a good buffer size 3072 works for Pi
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
 


def one_hot_encoder(labels):
  n_label = len(labels)
  n_unique_labels = len(np.unique(labels))
  one_hot_encoder = np.zeros((n_label, n_unique_labels))
  one_hot_encoder[np.arange(n_label), labels] = 1
  return one_hot_encoder

index_to_chord = {0: "Em",
                  1: "G"}

# #Reading the dataset with panda
# df = ps.read_csv("dataset_test.csv")
# X = df[df.columns[0:1023]].values
# y = df[df.columns[1023]].values
# X, y = shuffle(X, y, random_state = 1)
# encoder = LabelEncoder()
# encoder.fit(y)
# y = encoder.transform(y)
# Y = one_hot_encoder(y)

#train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.9, random_state = 415)

SR = SoundRecorder()
SR.setup()
raw_data_signal = SR.getAudio()
SR.close()

#DEFINING NEURAL NET
x = tf.placeholder(tf.float32, [None, len(raw_data_signal)])



W1 = tf.get_variable("W1", shape = [1023, 1500]) #1500 neurons
b1 = tf.get_variable("b1", shape = [1500])
W2 = tf.get_variable("W2", shape = [1500, 500]) #500 neurons
b2 = tf.get_variable("b2", shape = [500])
W3 = tf.get_variable("W3", shape = [500, 100]) #100 neurons
b3 = tf.get_variable("b3", shape = [100])
W4 = tf.get_variable("W4", shape = [100, 10]) #10 neurons
b4 = tf.get_variable("b4", shape = [10])
W5 = tf.get_variable("W5", shape = [10, 2])
b5 = tf.get_variable("b5", shape = [2])

y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
y2 = tf.nn.tanh(tf.matmul(y1, W2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
y = tf.nn.softmax(tf.matmul(y4, W5) + b5)



# Add ops to save and restore all the variables.
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "../models/adam_new/EGchords_adam_new.ckpt")

while(1):
  
    SR.setup()
    raw_data_signal = SR.getAudio()
    SR.close()
    raw_data_signal = np.array(raw_data_signal)
    #print(type(raw_data_signal))
        #print("Model restored.")
        # Check the values of the variables
    classification = sess.run(y, feed_dict = {x: [raw_data_signal]})
    if classification[0][0]>classification[0][1]:
        print("Em")
    else:
        print("G")

    
