#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:58:31 2018

@author: valeriepourquie
"""

import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn import datasets
from sklearn.model_selection import train_test_split

print(os.getcwd())
os.chdir('/Users/valeriepourquie/Documents/Master/DL/Project')
print(os.listdir())

#%% HIGGS training and validation set preprocessing 

df_train = pd.read_csv("HIGGS_training.csv")
df_train.head(5)

# Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and validation set
H_X_train = df_train.drop('Label', axis=1).as_matrix()
H_y_train = df_train['Label'].as_matrix()
H_X_train, H_X_val, H_y_train, H_y_val = train_test_split(H_X_train, H_y_train, test_size=0.2)

# Make labels binary 
H_y_val_bin = np.where(H_y_val=='b', 1, 0)
H_y_train_bin = np.where(H_y_train=='b', 1, 0)

#%% HIGGS test set

df_test = pd.read_csv("HIGGS_test.csv")
df_test.head(5)

# Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and validation set
H_X_test = df_train.drop('Label', axis=1).as_matrix()
H_y_test = df_train['Label'].as_matrix()

H_y_test_bin = np.where(H_y_test=='b', 1, 0)

#%% Not used

# Convert to tensor
H_X_train_tf = tf.convert_to_tensor(H_X_train, np.float32)
H_X_val_tf = tf.convert_to_tensor(H_X_val, np.float32)
H_y_train_tf = tf.convert_to_tensor(H_y_train_bin, np.float32)
H_y_val_tf = tf.convert_to_tensor(H_y_val_bin, np.float32)

H_train_dataset = tf.data.Dataset.from_tensor_slices((H_X_train,H_y_train_bin))
H_val_dataset = tf.data.Dataset.from_tensor_slices((H_X_val,H_y_val_bin))

sess = tf.InteractiveSession()  
print(H_X_train_tf.eval())
print(H_X_val_tf.eval())
print(H_y_train_tf.eval())
print(H_y_val_tf.eval())
sess.close()

#%% HIGGS test set preprocessing



#%% MNIST

mnist = fetch_mldata('MNIST original')

y = pd.Series(mnist.target).astype('int').astype('category')
X = pd.DataFrame(mnist.data)

mn_X_train, mn_X_val, mn_y_train, mn_y_val = train_test_split(X,y, test_size=0.2)
print(mn_X_train.shape)
print(mn_y_train.shape)
print(mn_X_val.shape)
print(mn_y_val.shape)


#%% show random image
ra = np.random.randint(1,len(X),1)
random_image = X.loc[ra,:]
random_label = y[ra]
plottable_image = np.reshape(random_image.values, (28, 28))
plt.imshow(plottable_image, cmap='gray_r')
plt.title('Digit Label: {}'.format(random_label))
plt.show()

