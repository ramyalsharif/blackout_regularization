import tensorflow as tf
from sigmoidBlackout import getActivePenalty

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

def get_regularization_penalty(weights, scale, percent_connections_kept, regularization_type):
    
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