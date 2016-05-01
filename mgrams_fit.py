#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This example showcases how simple it is to build image classification networks.
It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
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
from numpy import float32
from numpy import uint8
import random
from sklearn.cross_validation import StratifiedKFold

### Download and load MNIST data.

mnist = input_data.read_data_sets('MNIST_data')
# print(type(mnist.train.images))
# print(type(mnist.train.images[0]))
# print(len(mnist.train.images))
# print(mnist.train.images[0])
# print(len(mnist.train.images[0]))
# print(mnist.test.labels)
# print(type(mnist.train.labels[0]))
# print(type(mnist.train.images[0][0]))
# print(mnist.train.images[0])

temp = readMammograms.readData()
mgram_data = temp.data
mgram_labels = temp.labels
# mgram_data = []
# mgram_labels = []
# white = [255 for i in range(48*48)]
# black = [0 for i in range(48*48)]

# for i in range(50):
#     mgram_data.append(black)
# for i in range(50):
#     mgram_data.append(white)

# mgram_labels = [0 for i in range(50)]
# for i in range(50):
#     mgram_labels.append(1)

# mgram_data = [temp.data[i] for i in range(len(temp.labels)) if temp.labels[i] == 1 or temp.labels[i] == 2]
# mgram_labels = [temp.labels[i] for i in range(len(temp.labels)) if temp.labels[i] == 1 or temp.labels[i] == 2]
# print (len(mgram_data))
# print(len(mgram_labels))
# print(type(mgrams.data))
# print((mgrams.labels[0]))

# print(mgrams.labels[50])

### Linear classifier.

# classifier = skflow.TensorFlowLinearClassifier(
#     n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
# classifier.fit(mnist.train.images, mnist.train.labels)
# score = metrics.accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
# print('Accuracy: {0:f}'.format(score))

### Convolutional network

def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

def conv_model(X, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    X = tf.reshape(X, [-1, 48, 48, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5],
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5],
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        # 25 * 25 after max pooling twice
        h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
    # densely connected layer with 1024 neurons
    # h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu)
    return skflow.models.logistic_regression(h_fc1, y) #softmax?

# Prepare kfold training data splits
print(mgram_labels)
skf = StratifiedKFold(mgram_labels, n_folds = 9, shuffle = True)
training_data = []
test_data  = []
training_labels  = []
test_labels   = []
i = 0
for train_indices, test_indices in skf:
    # print("Train len: ", len(train_indices), "Test len: ", len(test_indices))
    train_batch = np.array([mgram_data[i] for i in train_indices])
    train_label = np.array([mgram_labels[i] for i in train_indices])
    test_batch = np.array([mgram_data[i] for i in test_indices])
    test_label = np.array([mgram_labels[i] for i in test_indices])
    if i == 0:
        print(train_label)
        print(test_label)
    # for index in train_index:
    #     train_batch.append(mgrams.data[index])
    #     train_label.append(mgrams.labels[index])

    training_data.append(np.ndarray(shape=(len(train_indices),48*48), dtype=float32))
    training_labels.append(np.ndarray(shape=(len(train_indices),), dtype=uint8))
    test_data.append(np.ndarray(shape=(len(test_indices),48*48), dtype=float32))
    test_labels.append(np.ndarray(shape=(len(test_indices),), dtype=uint8))

    for j in range(len(train_indices)):
        training_labels[i][j] = train_label[j]
        training_data[i][j] = train_batch[j]
    for j in range(len(test_indices)):
        test_labels[i][j] = test_label[j]
        test_data[i][j]   = test_batch[j]
    i += 1

print(training_data[0])
print(len(test_data[0]))


avg_accuracy = 0.0


# Training and predicting
classifier = skflow.TensorFlowEstimator(
    model_fn=conv_model, n_classes=3, steps=5000, #number of steps
    learning_rate=0.005)

# training_data = mgrams.data[:int((9 * len(mgrams.data) / 10))]
# test_data = mgrams.data[int((9 * len(mgrams.data) / 10)):]
# training_labels = mgrams.labels[:int((9 * len(mgrams.data) / 10))]
# test_labels = mgrams.labels[int(9 * len(mgrams.data) / 10):]
i = 0
classifier.fit(training_data[i], training_labels[i])
print("Fitted data")
results = classifier.predict(test_data[i])
print("Test labels")
print(test_labels[i])
print("Results:")
print (results, "\n")
accuracy = metrics.accuracy_score(test_labels[i], results)
print('Accuracy: {0:f}'.format(accuracy))
avg_accuracy += accuracy
precision = metrics.precision_score(test_labels[i], results)
print('Precision: {0:f}'.format(precision))
recall = metrics.recall_score(test_labels[i], results)
print('Recall: {0:f}'.format(recall))
print("Average Accuracy: ", avg_accuracy / 10, "\n")
print("Finished")
