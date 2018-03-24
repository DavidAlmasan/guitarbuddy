import pandas as ps 
import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



#
#
#
# using hidden layers that are larger than the input vector, also doing checks for different learning rates to see which one is the best 
# also using smaller batches and adam optimiser
#
#
#


def one_hot_encoder(labels):
	n_label = len(labels)
	n_unique_labels = len(np.unique(labels))
	one_hot_encoder = np.zeros((n_label, n_unique_labels))
	one_hot_encoder[np.arange(n_label), labels] = 1
	return one_hot_encoder




#Reading the dataset with panda
df = ps.read_csv("dataset.csv")
X = df[df.columns[0:1023]].values
y = df[df.columns[1023]].values
X, y = shuffle(X, y, random_state = 1)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encoder(y)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.30, random_state = 415)




#hyperparams
learning_rate = 0.0001 # one of you change this to 0.00001
epochs = 200
neurons = 1500 #number of neurons for first layer # initial configuration : 1500-300-75-10-2
accuracy_max = 0
accuracy_current = 0
batch_size = 500 #originally 550
validation_time = 20 #number of epochs needed for the loss to change significantly enough to stop training
#print(train_x.shape)


#storing the epoch, batch and accuracy
info_size = epochs * int(train_x.shape[0]/batch_size)      #train_x.shape[0] = number of examples in train
information = np.zeros((info_size, 3), dtype = np.float32)
chunk = np.zeros(3, dtype = np.float32)


x = tf.placeholder(tf.float32, [None, X[0].size])

#initialising weights and biases
W1 = tf.Variable(tf.truncated_normal([X[0].size, neurons], stddev = 0.1)) #2000 neurons
b1 = tf.Variable(tf.zeros([neurons]))
#W2 = tf.Variable(tf.truncated_normal([neurons, int(neurons/2)], stddev = 0.1)) #1000 neurons
#b2 = tf.Variable(tf.zeros([int(neurons/2)]))
#W3 = tf.Variable(tf.truncated_normal([int(neurons/2), int(neurons/4)], stddev = 0.1)) #500 neurons
#b3 = tf.Variable(tf.zeros([int(neurons/4)]))
W4 = tf.Variable(tf.truncated_normal([int(neurons), int(neurons/3)], stddev = 0.1)) #500 neurons
b4 = tf.Variable(tf.zeros([int(neurons/3)]))
W5 = tf.Variable(tf.truncated_normal([int(neurons/3), int(neurons/15)], stddev = 0.1)) #10 neurons
b5 = tf.Variable(tf.zeros([int(neurons/15)]))
W6 = tf.Variable(tf.truncated_normal([int(neurons/15), 2], stddev = 0.1))
b6 = tf.Variable(tf.zeros([2]))


#forward pass
y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
#y2 = tf.nn.tanh(tf.matmul(y1, W2) + b2)
#y3 = tf.nn.tanh(tf.matmul(y2, W3) + b3)
y4 = tf.nn.relu(tf.matmul(y1, W4) + b4)
y5 = tf.nn.relu(tf.matmul(y4, W5) + b5)
y = tf.nn.softmax(tf.matmul(y5, W6) + b6)

y_ = tf.placeholder(tf.float32, [None, 2])

#backwards pass

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))
train_step = tf.train.AdamOptimizer(learning_rate, beta1 = 0.95).minimize(cross_entropy)
init = tf.global_variables_initializer()
#saver = tf.train.Saver({"W1": W1, "W2": W2, "W3": W3, "W4": W4,"W5": W5, "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5})

sess = tf.Session()
sess.run(init)

percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
for percent in range(epochs):
	index = 0 #index to start the batch from
	print(percent)
	while index < train_x.shape[0]-batch_size -1: #not sure if -1 is needed, just making sure
		xs, ys = train_x[index: index + batch_size], train_y[index: index + batch_size] #training only in batches, for each epoch
		index += batch_size
		sess.run(train_step, feed_dict = {x: xs, y_:ys})
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy_value = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
		chunk = np.copy([percent+1, int(index/batch_size), accuracy_value])
		information[percent*(int(train_x.shape[0]/batch_size))+int(index/batch_size)-1] = chunk #that weird looking index in INFORMATION is just to din the place that we are in the training scheme
		# ###### logging to file
		# if percent == 0 and index == batch_size:
		# 	accuracy_over_epochs = open("accuracy_over_epochs.txt", 'w')
		# 	accuracy_over_epochs.write("Epoch: " + str(percent+1) + '\n')
		# 	accuracy_over_epochs.write("Batch number: " + str(int(index/batch_size)) + '\n')
		# 	accuracy_over_epochs.write("Accuracy: " + str(sess.run(accuracy, feed_dict={x: test_x, y_: test_y})) + '\n')
		# 	accuracy_over_epochs.write("------------------------------------------" + '\n')
		# else:
		# 	accuracy_over_epochs = open("accuracy_over_epochs.txt", 'a' )
		# 	accuracy_over_epochs.write("Epoch: " + str(percent+1) + '\n')
		# 	accuracy_over_epochs.write("Batch number: " + str(int(index/batch_size)) + '\n')
		# 	accuracy_over_epochs.write("Accuracy: " + str(sess.run(accuracy, feed_dict={x: test_x, y_: test_y})) + '\n')
		# 	accuracy_over_epochs.write("------------------------------------------" + '\n')




		print("Epoch: ", percent+1)
		print("Batch number: ", int(index/batch_size))
		print("Accuracy:", accuracy_value)
		#print("Cross entropy: ", cross_entropy)
		print("-----------------------------")
		# for i in percentages:
		# 	if percent/epochs == i:
		# 		print("Training at {}%".format(i*100))
		# 		print("------------------------------------------")
accuracy_vector = open("accuracy_vector.txt", 'w')
print(information)
for i in information:
	accuracy_vector = open("accuracy_vector.txt", 'a')
	accuracy_vector.write(str(i)+ '\n')
#save_path = saver.save(sess, "../models/training_new_structure_2/EGchords_training_new_structure_2.ckpt") #new structure - 0.9479 accuracy new structure 2 -
#print("Model saved in file:  \n", save_path)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
#print(train_x.shape[0])

