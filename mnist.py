from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

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


def create_model(X, num_layers, num_nodes):
  '''Creates a network architecture given the number of layers, and the number of nodes within each layer'''
  '''Returns a reference to a tensor that is dependent on x; it's not doing anything, apart from defining how the network looks like. '''

  # Constant variables, some to be changed to dynamic usage based on input
  num_inputs = 28 * 28
  num_classes = 10

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


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  num_layers = 10
  num_nodes = 784


  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y = create_model(x,num_layers,num_nodes)



  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
    if i % 100:
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