from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import tensorflow as tf

FLAGS = None


# 2: function that returns a tensorflow graph with

# variables:
# 1. number of layers.
# 2. nodes per layer.
# 3. regularizer (string).
# 4. regularization parameter.


def generate_weights(num_layers, num_inputs, num_hidden_nodes, num_classes):
    keys = []
    values = []

    ## Generating keys for weight dictionary
    for i in range(0, num_layers):

        keys.append('h' + str(i + 1))
        if (i == 0):
            values.append(tf.Variable(tf.random_normal([num_inputs, num_hidden_nodes])))
        else:
            values.append(tf.Variable(tf.random_normal([num_hidden_nodes, num_hidden_nodes])))

    ## Appending output weights
    values.append(tf.Variable(tf.random_normal([num_hidden_nodes, num_classes])))
    keys.append('out')

    return dict(zip(keys, values))


def generate_biases(num_layers, num_hidden_nodes, num_classes):
    keys = []
    values = []

    ## Generating keys for bias dictionary
    for i in range(0, num_layers):
        keys.append('b' + str(i + 1))
        values.append(tf.Variable(tf.random_normal([num_hidden_nodes])))

    ## Appending output layer
    values.append(tf.Variable(tf.random_normal([num_classes])))
    keys.append('out')

    return dict(zip(keys, values))

    # Function that divided the dataset in num_steps batches
def get_batches(data_x, data_y, num_steps):
    n_batches = np.ceil(len(data_x) / num_steps)
    x_batches = np.array_split(data_x, n_batches)
    y_batches = np.array_split(data_y, n_batches)

    print(y_batches[0].shape)
    return x_batches, y_batches


def create_model(X, num_layers, num_nodes, num_classes):
    '''Creates a network architecture given the number of layers, and the number of nodes within each layer'''
    '''Returns a reference to a tensor that is dependent on x; it's not doing anything, apart from defining how the network looks like. '''

    # Constant variables, some to be changed to dynamic usage based on input
    num_inputs = int(X.shape[1])
   
    # Generating weights and biases
    weights = generate_weights(num_layers, num_inputs, num_nodes, num_classes)
    biases = generate_biases(num_layers, num_nodes, num_classes)

    # For output
    # weights_out = weights
    # biases_out = biases

    # Generating first layer
    layer = tf.add(tf.matmul(X, weights['h1']), biases['b1'])

    last_weight = weights['out']
    last_bias = biases['out']

    # Removing first and last weights and biases from dictionary for simple iteration
    del weights['h1']
    del biases['b1']
    del weights['out']
    del biases['out']

    # Adding layers
    for weight, bias in zip(weights, biases):
        layer = tf.add(tf.matmul(layer, weights[weight]), biases[bias])

    # Adding last layer
    last_layer = tf.matmul(layer, last_weight) + last_bias

    return last_layer  # , weights_out, biases_out

def sigmoidBlackout(X,a,shift):
  #Computes sigmoid of `x` element-wise.
  #Specifically, `y = 1 / (1 + exp(-a*x))`.
  return 1 / (1 + tf.exp(-(X-shift)*a))

def CountActive(X):
    a = tf.Variable(400.0,tf.float32)
    shift = tf.Variable(0.03,tf.float32)
    return tf.reduce_sum(sigmoidBlackout(X,a,shift)+sigmoidBlackout(-X,a,shift))

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    num_layers = 10
    num_nodes = 784
    num_inputs = int(train_x.shape[1])
    num_steps = 100
    num_classes = 10

    # Initialize training data placeholder, training label placeholder, and model
    x = tf.placeholder(tf.float32, [None, num_inputs])
    y_ = tf.placeholder(tf.int64, [None])
    y = create_model(x, num_layers, num_nodes, num_classes)

    # Retrieve weights
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    

    # Define regularization
    target = tf.Variable(4.0,tf.float32)
    init_regularization = tf.norm(weights,1)/tf.size(weights,out_type= tf.float32)
    regularization = tf.abs(target-CountActive(weights))/tf.size(weights,out_type= tf.float32) * init_regularization

    # Define loss components
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y) 
    loss = cross_entropy + regularization
    train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss)

    
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    

    all_batches_x, all_batches_y = get_batches(train_x, train_y, num_steps)

    # Train
    for i in range(len(all_batches_x)):

        sess.run(train_step, feed_dict={x: all_batches_x[i], y_: all_batches_y[i]})
        # Test trained model
        if i % 10 == 0:
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)