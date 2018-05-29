import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import scipy.io as sio
from higgs_preprocessing import filterCol, logarithmic_momentum, logarithmic_weights
from sklearn import preprocessing

# Function that divides the dataset in num_steps batches
def get_batches(data_x, data_y, num_steps):
    n_batches = np.ceil(len(data_x) / num_steps)
    x_batches = np.array_split(data_x, n_batches)
    y_batches = np.array_split(data_y, n_batches)
    print(y_batches[0].shape)
    return x_batches, y_batches

def store_results(dataset,reguType, num_layers, num_nodes, num_steps,reguScale,percentOfConnectionsKept,accuracyValidation, accuracyTest):
    with open('AccuracyResults'+dataset+reguType+'.txt', "a") as myfile:
        myfile.write(reguType+','+str(num_layers)+','+str(num_nodes)+','+str(num_steps)+','+str(reguScale)+','+str(percentOfConnectionsKept)+','+str(accuracyValidation)+','+str(accuracyTest)+'\n')



def split_data(dataset, training_set_size):
    
    if dataset=='MNIST':
        mnist = input_data.read_data_sets("/tmp/data/")
        train_x = mnist.train.images
        train_y = mnist.train.labels
        
        # First we shuffle the data
        train_x, waste_x, train_y, waste_y = train_test_split(train_x, train_y, test_size = 0, shuffle=True)
        
        # Then we take a total training set size 
        train_x = train_x[:training_set_size]
        train_y = train_y[:training_set_size]

        valid_x = mnist.validation.images
        valid_y = mnist.validation.labels
        test_x = mnist.test.images
        test_y = mnist.test.labels
        
    elif dataset=='SVHN':
        train_data = sio.loadmat('train_32x32.mat')
        test_data = sio.loadmat('test_32x32.mat')
        
   
        # access to the dict
        train_x = train_data['X'].transpose()
        train_x = train_x.reshape(train_x.shape[0],3072)
        train_y = train_data['y'].flatten()
        train_y[train_y == 10] = 0
        
        print('Shapes:')
        print(train_x.shape)
        print(train_y.shape)
        
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2,shuffle=True)
 
        # access to the dict
        test_x = test_data['X'].transpose()
        test_x = test_x.reshape(test_x.shape[0],3072)
        
        test_y = test_data['y'].flatten()
        test_y[test_y == 10] = 0
        
        # Then we take a total training set size 
        train_x = train_x[:training_set_size]
        train_y = train_y[:training_set_size]
        
    
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
        
        # Then we take a total training set size 
        train_x = train_x[:training_set_size]
        train_y = train_y[:training_set_size]
        
            
    return train_x, train_y, valid_x, valid_y, test_x, test_y