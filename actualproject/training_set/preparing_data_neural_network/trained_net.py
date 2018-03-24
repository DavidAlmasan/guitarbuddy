import pandas as ps 
import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def one_hot_encoder(labels):
	n_label = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encoder = np.zeros((n_label, n_unique_labels))
	one_hot_encoder[np.arange(n_label), labels] = 1
	return one_hot_encoder


neurons = 500 #number of neurons for first layer

#Reading the dataset with panda
df = ps.read_csv("dataset_test.csv")
X = df[df.columns[0:1023]].values
y = df[df.columns[1023]].values
X, y = shuffle(X, y, random_state = 1)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encoder(y)

#train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.9, random_state = 415)

x = tf.placeholder(tf.float32, [None, X[0].size])

W1 = tf.get_variable("W1", shape = [1023, 500]) #500 neurons
b1 = tf.get_variable("b1", shape = [500])
W2 = tf.get_variable("W2", shape = [500, 100]) #100 neurons
b2 = tf.get_variable("b2", shape = [100])
W3 = tf.get_variable("W3", shape = [100, 50]) #50 neurons
b3 = tf.get_variable("b3", shape = [50])
W4 = tf.get_variable("W4", shape = [50, 10]) #10 neurons
b4 = tf.get_variable("b4", shape = [10])
W5 = tf.get_variable("W5", shape = [10, 2])
b5 = tf.get_variable("b5", shape = [2])

y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
y = tf.matmul(y4, W5) + b5

y_ = tf.placeholder(tf.float32, [None, 2])



# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "../models/decreasing_learning_rate/EGchords_decreasing_learning_rate.ckpt")
  print("Model restored.")
  # Check the values of the variables
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  print(sess.run(accuracy, feed_dict={x: X, y_: Y}))

