import tensorflow as tf


# Function for creating tensorFlow weight variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


# Function for creating tensorFlow bias variables
def bias_variable(shape):
    initial = tf.constant(float(0), shape=shape)
    return tf.Variable(initial)
