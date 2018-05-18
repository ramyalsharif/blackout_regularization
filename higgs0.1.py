from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os 
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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


def create_model(X, num_layers, num_nodes, num_inputs,num_classes):
    '''Creates a network architecture given the number of layers, and the number of nodes within each layer'''
    '''Returns a reference to a tensor that is dependent on x; it's not doing anything, apart from defining how the network looks like. '''

    # Constant variables, some to be changed to dynamic usage based on input
    
   

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

# Remove columns
def filterCol(dataFram):
    data_cols = dataFram.columns.tolist()
    
    for cols in data_cols:
        
        # Drop ID
        if cols == 'EventId':
            dataFram.drop([cols], axis = 1)
        
        # Drop columns that have missing values 
        test_missing = (dataFram[cols] == -999.000)
        if test_missing.sum() > 0:
            dataFram = dataFram.drop([cols], axis = 1)
            
    return dataFram

#
# Functions from https://github.com/cbracher69/Kaggle-Higgs-Boson-Challenge/blob/master/Higgs%20Linear-Gaussian%20Model%20Archive.ipynb
#
#
    
# Preparation - Turn momenta, weights into logarithms, normalize non-angular data
def logarithmic_momentum(dataFram):
    # Replace momentum data with its logarithm
    # I will add a small offset to avoid trouble with (rare) zero entries

    cols =  cols = list(dataFram.columns)
  
    for column in cols:
        if ((column.count('_phi') + column.count('eta')) == 0):
            # Select momentum features only:
            
            print(np.log(dataFram[column] + 1e-8))
            dataFram[column+'_log'] = np.log(dataFram[column] + 1e-8)
            dataFram = dataFram.drop(column, axis = 1)
            
    return dataFram

def normalize_data(dataFram):
    # Normalize all non-angular data
   
    cols =  cols = list(dataFram.columns)
    
    for column in cols:
        if ((column.count('_log') + column.count('_eta')) > 0):

            avg = dataFram[column].mean()
            dev = dataFram[column].std()

            dataFram[column] = ((dataFram[column] - avg) / dev)
                       
    return dataFram

def logarithmic_weights(dataFram):
    # For training set, split off scoring information, 
    # and add the logarithm of the weight as separate column.
    
    dataFram_outcome = dataFram[['Weight', 'Label']].copy()
    dataFram_outcome['log(Weight)'] = np.log(dataFram_outcome['Weight'])

    # Remove target information from data set

    dataFram = dataFram.drop(['Weight', 'Label'], axis = 1)
    
    return dataFram, dataFram_outcome

#
#
# End external functions
#
#%%
def main(_):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir)
    os.chdir('C:\\Users\\micha\\Documents\\deep learning\\project')
    print(os.listdir())

    df_train = pd.read_csv("HIGGS_training.csv")
    df_test = pd.read_csv("HIGGS_test.csv")
    
    # Filter columns
    df_train = filterCol(df_train)
    df_test = filterCol(df_test)
    # Normalize data, train and validation have extra weight-function
    # Test set does not contain weights
    #df_train = logarithmic_momentum(df_train)
    df_train = normalize_data(df_train)
    #df_test = logarithmic_momentum(df_test)
    df_test = normalize_data(df_test)
        #H_X_train = logarithmic_weights(df_train)

    # Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and validation set
    H_X_train = df_train.drop(['Label','Weight'], axis=1).as_matrix()
    #drop weights aswell
   # H_X_train = H_X_train.drop('Weight', axis=1).as_matrix()
    H_y_train = df_train['Label'].as_matrix()
   
    #split data
    H_X_train, H_X_val, H_y_train, H_y_val = train_test_split(H_X_train, H_y_train, test_size=0.2)
    
    # Convert test Dataframe into NumPy array
    max_abs_scaler = preprocessing.MaxAbsScaler()
    H_X_test = max_abs_scaler.fit_transform(df_test)

    # Make labels binary 
    H_y_val_bin = np.where(H_y_val=='b', 1, 0)
    H_y_train_bin = np.where(H_y_train=='b', 1, 0)
    
    
    #scale data between 0 and 1
    max_abs_scaler = preprocessing.MaxAbsScaler()
    H_X_train = max_abs_scaler.fit_transform(H_X_train)
    max_abs_scaler = preprocessing.MaxAbsScaler()
    H_X_val = max_abs_scaler.fit_transform(H_X_val)  
    # Input Higgs data
    train_x = H_X_train
    train_y = H_y_train_bin
    num_layers = 5
    num_nodes = 32
    num_classes = 2
    num_input= 20

    # Create the model
    x = tf.placeholder(tf.float32, [None, num_input])
    y = create_model(x, num_layers, num_nodes,num_input,num_classes)

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
    num_steps = 100

    all_batches_x, all_batches_y = get_batches(train_x, train_y, num_steps)
    # Train
    for i in range(len(all_batches_x)):
        sess.run(train_step, feed_dict={x: all_batches_x[i], y_: all_batches_y[i]})
        #Test trained model
        if i % 10 == 0:
            print(sess.run(accuracy, feed_dict={x: H_X_val, y_: H_y_val_bin}))
    
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)