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



#Reading the dataset with panda
df = ps.read_csv("dataset.csv")
X = df[df.columns[0:1024]].values
y = df[df.columns[60]].values
X, y = shuffle(X, y, random_state = 1)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encoder(y)




train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.10, random_state = 415)

learning_rate = 0.25
epochs = 10000
neurons = 100 #number of neurons per layer

x = tf.placeholder(tf.float32, [None, X[0].size])

W1 = tf.Variable(tf.truncated_normal([X[0].size, neurons], stddev = 0.1))
b1 = tf.Variable(tf.zeros([neurons]))
W2 = tf.Variable(tf.truncated_normal([neurons, neurons], stddev = 0.1))
b2 = tf.Variable(tf.zeros([neurons]))
W3 = tf.Variable(tf.truncated_normal([neurons, neurons], stddev = 0.1))
b3 = tf.Variable(tf.zeros([neurons]))
W4 = tf.Variable(tf.truncated_normal([neurons, neurons], stddev = 0.1))
b4 = tf.Variable(tf.zeros([neurons]))
W5 = tf.Variable(tf.truncated_normal([neurons, 2], stddev = 0.1))
b5 = tf.Variable(tf.zeros([2]))

y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
y = tf.matmul(y4, W5) + b5

y_ = tf.placeholder(tf.float32, [None, 2])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
saver = tf.train.Saver({"W1": W1, "W2": W2, "W3": W3, "W4": W4,"b1": b1, "b2": b2, "b3": b3, "b4": b4})
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run() #initialise variables

sess = tf.Session()
sess.run(init)

percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
for percent in range(epochs):
	sess.run(train_step, feed_dict = {x: train_x, y_:train_y})
	for i in percentages:
		if percent/epochs == i:
			print("Training at {}%".format(i*100))
save_path = saver.save(sess, "./models/sonar/sonar.ckpt")
print("Model saved in file:  \n", save_path)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

