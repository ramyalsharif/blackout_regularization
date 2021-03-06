# -*- coding: utf-8 -*-

""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
from sigmoidBlackout import getActivePenalty
from sklearn.model_selection import train_test_split
import random
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

# Blackout Parameters
# 'HIGGS' or 'MNIST'
dataset='MNIST'
#    available regu types 'None','L1','L2','Blackout'
regularization_type='Blackout'
number_of_tests = 100

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def storeResults(dataset,reguType,reguScale, percentOfConnectionsKept, accuracyValidation, accuracyTest):
    with open(dataset+reguType+'.txt', "a") as myfile:
        myfile.write(reguType+','+str(reguScale)+','+str(percentOfConnectionsKept)+','+str(accuracyValidation)+','+str(accuracyTest)+'\n')

def get_regularization_penalty(weights, scale, percent_connections_kept):
    
    blackout_weights=None
    
    if regularization_type=='L1':
        regularizer = tf.contrib.layers.l1_regularizer(scale=scale, scope=None)
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
        
    elif regularization_type=='L2':
        regularizer = tf.contrib.layers.l2_regularizer(scale=scale, scope=None)
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, weights)
    
    elif regularization_type=='Blackout':
        regularizer = tf.contrib.layers.l1_regularizer(scale=scale, scope=None)
        regularizationL1 = tf.contrib.layers.apply_regularization(regularizer, weights)
        
        
        for w in weights:
            if not(w.shape.__ne__([])==False):
                if blackout_weights==None:
                    blackout_weights=tf.reshape(w, [-1])
                else:
                    blackout_weights=tf.concat([blackout_weights,tf.reshape(w, [-1])],axis=0)
        targetNumberOfWeights=blackout_weights.shape[0].value*scale
        penaltyNumOfActive=getActivePenalty(blackout_weights,targetNumberOfWeights)
        regularization_penalty=tf.cond(penaltyNumOfActive>0, lambda: penaltyNumOfActive*regularizationL1*100, lambda: tf.abs(penaltyNumOfActive)*targetNumberOfWeights*100)
    else:
        regularization_penalty=tf.constant(0.0)
    
    return regularization_penalty, blackout_weights

def split_data(data):
    
    if dataset=='MNIST':
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        train_x = mnist.train.images
        train_y = mnist.train.labels
        train_x, waste_x, train_y, waste_y = train_test_split(train_x, train_y, test_size=0.4, shuffle=True)

        valid_x = mnist.validation.images
        valid_y = mnist.validation.labels
        test_x = mnist.test.images
        test_y = mnist.test.labels
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def get_batches(data_x, data_y, num_steps):
    n_batches = np.ceil(len(data_x) / num_steps)
    x_batches = np.array_split(data_x, n_batches)
    y_batches = np.array_split(data_y, n_batches)
    print(y_batches[0].shape)
    return x_batches, y_batches

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
with tf.device('/gpu:0'):
    
    # Getting the appropriate dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(dataset)
    
    # Resetting the graph incase of multiple runs on the same console
    #tf.reset_default_graph()
    
    regularization_list = {'None','Blackout','L1','L2'}
    
    for regularization_type in regularization_list:
        for i in range(number_of_tests):
            
            num_steps = random.choice([50,100,150,200])
            regularization_scale = random.choice([0.01,0.005,0.001,0.0005])
            percent_connections_kept=random.choice([0.9,0.95,0.85])
            
            print('Test No. '+str(i)+'/'+str(number_of_tests))
            print('Parameters: '+regularization_type +',scale: ' + str(regularization_scale)+', percent kept:'+str(percent_connections_kept))
                
            
            # Construct model
            logits = conv_net(X, weights, biases, keep_prob)
            prediction = tf.nn.softmax(logits)
            
            # Retrieving weights and defining regularization penalty  
            trainable_weights=tf.trainable_variables()
            regularization_penalty, blackout_weights = get_regularization_penalty(trainable_weights, regularization_scale, percent_connections_kept)    
                
            
            # Define loss and optimizer
            cross=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=Y))
            loss_op = cross+regularization_penalty
            train_op = tf.train.RMSPropOptimizer(0.001).minimize(loss_op)
            
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            # Initialize the variables (i.e. assign their default value)
            init = tf.global_variables_initializer()
            
            # Start training
            with tf.Session() as sess:
            
                # Run the initializer
                sess.run(init)
                
                numOfBatches=50
                all_batches_x, all_batches_y = get_batches(train_x, train_y, numOfBatches)
        
            
                for step in range(1, num_steps+1):
                    
                    # Randomly pick batch
                    randomPick=random.randint(0,numOfBatches)
                    currentBatchX=all_batches_x[randomPick]
                    currentBatchY=all_batches_y[randomPick]  
                    
                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={X: currentBatchX, Y: currentBatchY, keep_prob: 0.8})
                    if step % display_step == 0 or step == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: valid_x,
                                                                             Y: valid_y,
                                                                             keep_prob: 1.0})
                        print("Step " + str(step) + ", Minibatch Loss= " + \
                              "{:.4f}".format(loss) + ", Training Accuracy= " + \
                              "{:.3f}".format(acc))
            
                print("Optimization Finished!")
            
                accuracyTest=sess.run(accuracy, feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
                    
            
                accuracyVal=sess.run(accuracy, feed_dict={X: valid_x, Y: valid_y, keep_prob: 1.0})
                storeResults(dataset,regularization_type, regularization_scale,percent_connections_kept,accuracyVal, accuracyTest)
                print('Accuracy Val: '+str(accuracyVal)+' , Accuracy Test: '+str(accuracyTest))