"""
Created by following TensorFlow tutorial on how to create a CNN
https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import skflow
import readMammograms
import numpy as np

#weights and biases
#initialize w/ small amount of noise to avoid symmetry
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#pooling and convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Vanilla 2x2
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#TODO normalize the pixel values -> z-scores ???

sess = tf.InteractiveSession()
mgrams = readMammograms.readData()
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(len(mnist.train.next_batch(50)[0]))
# print(len(mnist.train.next_batch(50)[1]))

# print(np.array_str(mnist.train.images[0]))
# print(type(mnist.train.images[0][0]))
# print(len(mnist.train.images[0]))
# print (np.array_str(mgrams.data[0]))
# print (len(mgrams.data))
# print (type(mgrams.data))
# print (type(mgrams.data[0]))


print(np.array_str(mgrams.data))

#placeholder, ask Tensorflow to be able to input
#any number of mammograms flattened into array of float32 of
# size of 1024*1024
x = tf.placeholder(tf.float32, shape=[None, 100 * 100])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

#model parameters
#vector of 1024^2 -> 3 classes
W = tf.Variable(tf.zeros([100 * 100, 3]))

b = tf.Variable(tf.zeros([3]))

# sess.run(tf.initialize_all_variables())


#first layer: convolution -> max pooling
#5x5 patch, 1 input channel, 32 output features
W_conv1 = weight_variable([5, 5, 1, 32])
#bias vector
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,100,100,1])

# convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Layer 2: 32 features -> 64 features
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Now that the image size has been reduced to 256x256, we add a fully-connected
# layer with 1024 neurons to allow processing on the entire image. We reshape
# the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

# Dropout layer to reduce overfitting
# Probability that neuron's output is kept during dropout
h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Layer???
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# print(y_conv)

# Train and Evaluate
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.initialize_all_variables())

# for i in range(20000):
for i in range(100):
  batch = (mgrams.data, mgrams.labels)
  # if i%100 == 0:
  #   train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
  #   print("step %d, training accuracy %g"%(i, train_accuracy))

  print("let's train!")
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  print("it trained")

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))









