from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import random
import numpy as np

import tensorflow as tf
from data_processing import split_data, get_batches, store_results
from model_functions import create_model, get_regularization_penalty

# 'HIGGS' or 'MNIST'
dataset='MNIST'
#    available regu types 'None','L1','L2','Blackout'
regularization_type='None'
numOfTests=50
training_set_size = 5000


# variables:
# 1. number of layers.
# 2. nodes per layer.
# 3. regularizer (string).
# 4. regularization parameter.    


def main(_):
    with tf.device('/gpu:0'):
    
        dataset_sizes = np.linspace(2500,55000, num=22)
        for size in dataset_sizes:
            # Getting the appropriate dataset
            train_x, train_y, valid_x, valid_y, test_x, test_y = split_data(dataset, size)
    
            # Resetting the graph incase of multiple runs on the same console
            tf.reset_default_graph()
            for i in range(numOfTests):
                num_layers = random.choice([5,6,7,8,9,10])
                num_nodes = random.choice([200,400,600])
                num_inputs = int(train_x.shape[1])
                num_steps = random.choice([50,100,150,200])
                regularization_scale = random.choice([0.01,0.005,0.001,0.0005])
                percent_connections_kept=random.choice([0.9,0.95,0.85])
                num_classes = len(np.unique(train_y))
                
                print('Test No. '+str(i)+'/'+str(numOfTests))
                print('Parameters: ' + str(size) + ','+regularization_type+','+str(num_layers)+','+str(num_nodes)+','+str(num_steps)+','+str(regularization_scale)+','+str(percent_connections_kept))
                
                # Create the model
                x = tf.placeholder(tf.float32, [None, num_inputs])
                y= create_model(x, num_layers, num_nodes, num_classes)
        
                # Define loss and optimizer
                y_ = tf.placeholder(tf.int64, [None])
                
                # Retrieving weights and defining regularization penalty  
                weights=tf.trainable_variables()
                regularization_penalty, blackout_weights = get_regularization_penalty(weights, regularization_scale, percent_connections_kept, regularization_type)    
                
                # Defining loss and optimizer
                cross=tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
                loss = cross+regularization_penalty
                train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss)
                
                # Evaluate Model
                correct_prediction = tf.equal(tf.argmax(y, 1), y_)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                # Initializing session
                sess = tf.InteractiveSession(config=config);
                tf.global_variables_initializer().run()
        
        
                # Train
#                PercentageOfConnOff=[]
#                LossFunctionRegu=[]
#                LossFunctionCrossTrain=[]
#                LossFunctionCrossValid=[]
#        
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
#                            if regularization_type=='Blackout':
#                                currentWeights=sess.run(blackout_weights)
#                                part1=currentWeights>-0.01
#                                part2=currentWeights<0.01
#                                turnedOff=np.sum(np.logical_and(part1,part2))
#                                TotalNumOfWeights=float(currentWeights.shape[0])
#                                LossFunctionCrossTrain.append(sess.run(cross, feed_dict={x: train_x, y_: train_y}))
#                                LossFunctionCrossValid.append(sess.run(cross, feed_dict={x: valid_x, y_: valid_y}))
#                                LossFunctionRegu.append(sess.run(regularization_penalty))
#                                PercentageOfConnOff.append((TotalNumOfWeights-turnedOff)/TotalNumOfWeights)
                #if regularization_type=='Blackout':
                #    fig = plt.figure()
                #    ax1 = fig.add_subplot(1, 2, 1)
                #   ax2 = fig.add_subplot(1, 2, 2)
                #    ax1.plot(PercentageOfConnOff)
                #    ax2.plot(LossFunctionCrossTrain,label='Cross-Entropy Train')
                #    ax2.plot(LossFunctionCrossValid,label='Cross-Entropy Validation')
                #    ax2.plot(LossFunctionRegu,label='Regularization')
                #    ax2.legend()
                #    fig.show()
                accuracyVal=sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
                accuracyTest=sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
                tf.reset_default_graph()
                store_results(dataset,regularization_type, num_layers, num_nodes, num_steps,regularization_scale,percent_connections_kept,accuracyVal, accuracyTest, size)
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