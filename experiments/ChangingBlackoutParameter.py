from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import random
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sigmoidBlackout import getActivePenalty
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

FLAGS = None
# 'HIGGS' or 'MNIST'
dataset='MNIST'
#    available regu types 'None','L1','L2','Blackout'
regularization_type='Blackout'
numOfTests=50


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

# Remove columns
def filterCol(dataFram):
    data_cols = dataFram.columns.tolist()
    
    for cols in data_cols:
        
        # Drop ID
        if cols == 'EventId' or cols == 'KaggleSet' or cols=='KaggleWeight' or cols=='Weight':
            dataFram=dataFram.drop([cols], axis = 1)
            continue
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
            if not(column=='Label' or column=='Weight'):
                dataFram[column+'_log'] = np.log(dataFram[column] + 1e-8)
                dataFram = dataFram.drop(column, axis = 1)
            
    return dataFram


def logarithmic_weights(dataFram):
    # For training set, split off scoring information, 
    # and add the logarithm of the weight as separate column.
    dataFram_outcome = dataFram['Label'].copy()
    # Remove target information from data set
    dataFram = dataFram.drop(['Label'], axis = 1)
    return pd.concat((dataFram, dataFram_outcome),axis=1)


def storeResults(dataset,reguType, num_layers, num_nodes, num_steps,reguScale,percentOfConnectionsKept,accuracyValidation, accuracyTest, blackout):
    with open('AccuracyResults'+dataset+reguType+'blackout' + str(blackout) + '.txt', "a") as myfile:
        myfile.write(reguType+','+str(num_layers)+','+str(num_nodes)+','+str(num_steps)+','+str(reguScale)+','+str(percentOfConnectionsKept)+','+str(accuracyValidation)+','+str(accuracyTest)+'\n')

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
        mnist = input_data.read_data_sets("/tmp/data/")
        train_x = mnist.train.images
        train_y = mnist.train.labels
        train_x, waste_x, train_y, waste_y = train_test_split(train_x, train_y, test_size=0.4, shuffle=True)

        valid_x = mnist.validation.images
        valid_y = mnist.validation.labels
        test_x = mnist.test.images
        test_y = mnist.test.labels
    else:
        df_train = pd.read_csv("atlas-higgs-challenge-2014-v2.csv")

        # Filter columns
        df_train = filterCol(df_train)

        # Normalize data, train and validation have extra weight-function
        # Test set does not contain weights
        df_train = logarithmic_momentum(df_train)
        df_train = logarithmic_weights(df_train)

        # Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and validation set
        H_X_train = df_train.drop('Label', axis=1).as_matrix()
        H_y_train = df_train['Label'].as_matrix()
        max_abs_scaler = preprocessing.MaxAbsScaler()
        H_X_train = max_abs_scaler.fit_transform(H_X_train)

        # Make labels binary
        H_y_train=np.where(H_y_train=='b', 1, 0)
        train_x, test_x, train_y, test_y = train_test_split(H_X_train, H_y_train, test_size=0.5,shuffle=True)
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2,shuffle=True)
            
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def main(_):
    #with tf.device('/gpu:0'):
    
    # Getting the appropriate dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(dataset)

    # Resetting the graph incase of multiple runs on the same console
    tf.reset_default_graph()
    for percent_connections_kept in np.arange(0.85, 1, 0.05): 
        for i in range(numOfTests):
            num_layers = random.choice([5,6,7,8,9,10])
            num_nodes = random.choice([200,400,600])
            num_inputs = int(train_x.shape[1])
            num_steps = random.choice([50,100,150,200])
            regularization_scale = random.choice([0.01,0.005,0.001,0.0005])
            num_classes = 10
            print('Test No. '+str(i)+'/'+str(numOfTests))
            print('Parameters: '+regularization_type+','+str(num_layers)+','+str(num_nodes)+','+str(num_steps)+','+str(regularization_scale)+','+str(percent_connections_kept))
            
            # Create the model
            x = tf.placeholder(tf.float32, [None, num_inputs])
            y= create_model(x, num_layers, num_nodes, num_classes)
    
            # Define loss and optimizer
            y_ = tf.placeholder(tf.int64, [None])
            
            # Retrieving weights and defining regularization penalty  
            weights=tf.trainable_variables()
            regularization_penalty, blackout_weights = get_regularization_penalty(weights, regularization_scale, percent_connections_kept)    
            
            # Defining loss and optimizer
            cross=tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
            loss = cross+regularization_penalty
            train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss)
            
            # Evaluate Model
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            # Initializing session
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
    
    
            # Train
            PercentageOfConnOff=[]
            LossFunctionRegu=[]
            LossFunctionCrossTrain=[]
            LossFunctionCrossValid=[]
    
            numOfBatches=50
            all_batches_x, all_batches_y = get_batches(train_x, train_y, numOfBatches)
    
            # Train
            for i in range(num_steps):
                randomPick=random.randint(0,numOfBatches)
                currentBatchX=all_batches_x[randomPick]
                currentBatchY=all_batches_y[randomPick]                
                sess.run(train_step, feed_dict={x: currentBatchX, y_: currentBatchY})
                # Test trained model
                if i % 20 == 1:
                    print('Accuracy: '+str(sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})))
                    if regularization_type=='Blackout':
                        currentWeights=sess.run(blackout_weights)
                        part1=currentWeights>-0.01
                        part2=currentWeights<0.01
                        turnedOff=np.sum(np.logical_and(part1,part2))
                        TotalNumOfWeights=float(currentWeights.shape[0])
                        LossFunctionCrossTrain.append(sess.run(cross, feed_dict={x: train_x, y_: train_y}))
                        LossFunctionCrossValid.append(sess.run(cross, feed_dict={x: valid_x, y_: valid_y}))
                        LossFunctionRegu.append(sess.run(regularization_penalty))
                        PercentageOfConnOff.append((TotalNumOfWeights-turnedOff)/TotalNumOfWeights)
            if regularization_type=='Blackout':
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                ax1.plot(PercentageOfConnOff)
                ax2.plot(LossFunctionCrossTrain,label='Cross-Entropy Train')
                ax2.plot(LossFunctionCrossValid,label='Cross-Entropy Validation')
                ax2.plot(LossFunctionRegu,label='Regularization')
                ax2.legend()
                fig.show()
            accuracyVal=sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            accuracyTest=sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            tf.reset_default_graph()
            storeResults(dataset,regularization_type, num_layers, num_nodes, num_steps,regularization_scale,percent_connections_kept,accuracyVal, accuracyTest, percent_connections_kept)
            print('Accuracy Val: '+str(accuracyVal)+' , Accuracy Test: '+str(accuracyTest))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)