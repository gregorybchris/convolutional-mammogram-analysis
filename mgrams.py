from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import skflow
import readMammograms

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

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



#placeholder, ask Tensorflow to be able to input
#any number of mammograms flattened into array of float32 of
# size of 1024*1024
x = tf.placeholder(tf.float32, [None, 1048576])

#model parameters
#vector of 1024^2 -> 3 classes
W = tf.Variable(tf.zeros([1048576, 3]))

b = tf.Variable(tf.zeros([3]))

mgrams = readMammograms.readData()


#first layer: convolution -> max pooling
#5x5 patch, 1 input channel, 32 output features
W_conv1 = weight_variable([5, 5, 1, 32])
#bias vector
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,1024,1024,1])

# convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)









