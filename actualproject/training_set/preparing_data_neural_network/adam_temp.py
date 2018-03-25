import pandas as ps 
import numpy as np 
import tensorflow as tf 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



# idk why but its the best working as of now together with adam optimiser
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

def validation_error_mean(tensor):
	mean = 0
	for i in tensor:
		mean += i
	mean = mean/tensor.shape[0]





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
learning_rate = 0.00001 # one of you change this to 0.00001
epochs = 500
neurons = 1500 #number of neurons for first layer # initial configuration : 1500-300-75-10-2
accuracy_max = 0
accuracy_current = 0
batch_size = 550 #originally 550
validation_time = 5 #number of epochs needed for the loss to change significantly enough to stop training
check_epoch = 0 #will be incremented by validation_time and check everytime
validation_error_vector = []
#print(train_x.shape)


#storing the epoch, batch and accuracy
info_size = epochs * int(train_x.shape[0]/batch_size)      #train_x.shape[0] = number of examples in train
information = np.zeros((info_size, 3), dtype = np.float32)
chunk = np.zeros(3, dtype = np.float32)


x = tf.placeholder(tf.float32, [None, X[0].size])

#initialising weights and biases
W1 = tf.Variable(tf.truncated_normal([X[0].size, neurons], stddev = 0.1)) #1500 neurons
b1 = tf.Variable(tf.zeros([neurons]))
W2 = tf.Variable(tf.truncated_normal([neurons, int(neurons/3)], stddev = 0.1)) #500 neurons
b2 = tf.Variable(tf.zeros([int(neurons/3)]))
W3 = tf.Variable(tf.truncated_normal([int(neurons/3), int(neurons/15)], stddev = 0.1)) #100 neurons
b3 = tf.Variable(tf.zeros([int(neurons/15)]))
W4 = tf.Variable(tf.truncated_normal([int(neurons/15), int(neurons/150)], stddev = 0.1)) #10 neurons
b4 = tf.Variable(tf.zeros([int(neurons/150)]))
W5 = tf.Variable(tf.truncated_normal([int(neurons/150), 2], stddev = 0.1))
b5 = tf.Variable(tf.zeros([2]))


#forward pass
y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
y2 = tf.nn.tanh(tf.matmul(y1, W2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
y = tf.nn.softmax(tf.matmul(y4, W5) + b5)

y_ = tf.placeholder(tf.float32, [None, 2])

#backwards pass

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))
train_step = tf.train.AdamOptimizer(learning_rate, beta1 = 0.95).minimize(cross_entropy)
init = tf.global_variables_initializer()
saver = tf.train.Saver({"W1": W1, "W2": W2, "W3": W3, "W4": W4,"W5": W5, "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5}) #if smaller than previous continue training and log values of tensors


sess = tf.Session()
sess.run(init)

percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
for percent in range(epochs):
	index = 0 #index to start the batch from
	print("Epoch: ", percent)
	while index < train_x.shape[0]-batch_size -1: #not sure if -1 is needed, just making sure
		xs, ys = train_x[index: index + batch_size], train_y[index: index + batch_size] #training only in batches, for each epoch
		index += batch_size
		sess.run(train_step, feed_dict = {x: xs, y_:ys})
	if percent == check_epoch:
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_)) #check validation entropy 
		validation_error_vector.append(float(sess.run(cross_entropy, feed_dict = {x: test_x, y_:test_y}))) #if bigger than previous STOP TRAINING
		if validation_error_vector[-1] == np.min(validation_error_vector):
			save_path = saver.save(sess, "../models/adam_new/EGchords_adam_new.ckpt") #new structure - 0.9479 accuracy new structure 2 -
		check_epoch += validation_time
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy_value = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
		#print("Epoch: ", percent)
		#print("Batch number: ", int(index/batch_size))
		print("Accuracy:", accuracy_value)
		print("Validation error:", validation_error_vector[-1])
		print("-----------------------------")
		# for i in percentages:
		# 	if percent/epochs == i:
		# 		print("Training at {}%".format(i*100))
		# 		print("------------------------------------------")

#save_path = saver.save(sess, "../models/training_new_structure_2/EGchords_training_new_structure_2.ckpt") #new structure - 0.9479 accuracy new structure 2 -
print("Model saved in file:  \n", save_path)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
print("Validation error vector", '\n', validation_error_vector)
#print(train_x.shape[0])

