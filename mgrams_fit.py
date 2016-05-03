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

# mnist = input_data.read_data_sets('MNIST_data')
# # print(type(mnist.train.images))
# # print(type(mnist.train.images[0]))
# # print(len(mnist.train.images))
# # print(mnist.train.images[0])
# # print(len(mnist.train.images[0]))
# # print(mnist.test.labels)
# # print(type(mnist.train.labels[0]))
# # print(type(mnist.train.images[0][0]))
# # print(mnist.train.images[0])
##############################################################################
"""
Load mammogram data

"""
print("Loading in Mammogram data...")
mgrams = readMammograms.readData()
mgram_data = mgrams.data
mgram_labels = mgrams.labels
print("Mammogram data loaded.\n")
### Linear classifier.

# classifier = skflow.TensorFlowLinearClassifier(
#     n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
# classifier.fit(mnist.train.images, mnist.train.labels)
# score = metrics.accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
# print('Accuracy: {0:f}'.format(score))

##############################################################################
"""
Convolutional Network Model

"""


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
        #DNN with 1024 hidden layers, and dropout of 0.5 probability.
        h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
        #  """DNN with 64,128,64 hidden layers, and dropout of 0.5 probability."""
        # h_fc1 = skflow.ops.dnn(h_pool2_flat, [64,128,64], activation=tf.nn.relu, keep_prob=0.5)

    return skflow.models.logistic_regression(h_fc1, y) #softmax?


##############################################################################
"""
Prepare kfold training data splits

"""

folds = 8
print("Performing ", folds, "-fold validation")
skf = StratifiedKFold(mgram_labels, n_folds = folds, shuffle = True)
training_data    = []
test_data        = []
training_labels  = []
test_labels      = []
i = 0
for train_indices, test_indices in skf:
    # print("Train len: ", len(train_indices), "Test len: ", len(test_indices))
    train_batch = np.array([mgram_data[i] for i in train_indices])
    train_label = np.array([mgram_labels[i] for i in train_indices])
    test_batch = np.array([mgram_data[i] for i in test_indices])
    test_label = np.array([mgram_labels[i] for i in test_indices])
    # if i == 0:
    #     print(train_label)
    #     print(test_label)
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

print(folds, "-fold validation prepared.\n")


##############################################################################
"""
Train and test model

"""

lr    = 0.003
steps = 5000

classifier = skflow.TensorFlowEstimator(
    model_fn=conv_model, n_classes=3, steps=steps, #number of steps
    learning_rate=lr)

print("Training convolutional NN with ", steps, " steps with ", lr, " learning rate.")
i = 0
classifier.fit(training_data[i], training_labels[i])
print("Fitted data")
results = classifier.predict(test_data[i])
print("Expected labels:                 (0 = Normal, 1 = Benign, 2 = Cancerous)")
print(test_labels[i])
print("\nPredicted labels:")
print (results)

# Training Accuracy (check for overfitting)
training_results = classifier.predict(training_data[i])
training_accuracy = metrics.accuracy_score(training_labels[i], training_results)
print("\nTraining Predicted labels:")
print (training_results, "\n")
# Accuracy
accuracy = metrics.accuracy_score(test_labels[i], results)

print("Printing metrics ================")
print('Accuracy          : {0:f}'.format(accuracy))
print('Training Accuracy : {0:f}'.format(training_accuracy))
# Precision
precision = metrics.precision_score(test_labels[i], results, labels=[2]) #only calculate for malignant case
print('Precision         : {0:f}'.format(precision))
# Recall ***
recall = metrics.recall_score(test_labels[i], results, labels=[2])       #only calculate for malignant case
print('Recall            : {0:f}'.format(recall))
f1_score = metrics.f1_score(test_labels[i], results, labels=[2])
print('F1 Score          : {0:f}'.format(f1_score))
print("Finished.")
