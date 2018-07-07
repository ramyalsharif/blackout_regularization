import tensorflow as tf

def sigmoidBlackout(X,a,shift):
  #Computes sigmoid of `x` element-wise.
  #Specifically, `y = 1 / (1 + exp(-a*x))`.
  return 1 / (1 + tf.exp(-(X-shift)*a))
#
#def CountActive(X):
#    a =400.0
#    shift = 0.03
#    return tf.reduce_sum(sigmoidBlackout(X,a,shift)+sigmoidBlackout(-X,a,shift))

def CountActive(X):
    value=0.02
    part1=tf.greater(X,-value)
    part2=tf.greater(value,X)
    active=tf.logical_not(tf.logical_and(part1,part2))
    return tf.reduce_sum(tf.cast(active, tf.float32))

def getActivePenalty(weights,target):
    result=(CountActive(weights)-tf.cast(target,tf.float32))/tf.cast(target,tf.float32)
    return result

#def getRegularizationLossBlackout(weights,reguScale,target):
#    regularization_penalty = reguScale*tf.norm( weights,1)
#    result=tf.abs(target-CountActive(weights))/tf.size(weights,out_type= tf.float32)*regularization_penalty # that would be the current regu
#    return result
#weights=tf.zeros([20, 10], tf.float32)
#target = tf.Variable(4.0,tf.float32)

#regularization_penalty = tf.norm( weights,1)/tf.size(weights,out_type= tf.float32)
#result=tf.abs(target-CountActive(weights))/tf.size(weights,out_type= tf.float32)*regularization_penalty # that would be the current regu


#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(result))