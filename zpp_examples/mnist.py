from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import tensorflow as tf

conv_bin_module = tf.load_op_library('/home/osboxes/tensorflow_source/tensorflow/tensorflow/core/user_ops/conv_bin.so')

@ops.RegisterGradient("BinaryConv2D")
def _Conv2DGrad(op, grad):
  return [nn_ops.conv2d_backprop_input(
      array_ops.shape(op.inputs[0]), tf.sign(op.inputs[1]), grad, op.get_attr("strides"),
      op.get_attr("padding"), op.get_attr("use_cudnn_on_gpu"),
      op.get_attr("data_format")),
          nn_ops.conv2d_backprop_filter(op.inputs[0],
                                        array_ops.shape(op.inputs[1]), grad,
                                        op.get_attr("strides"),
                                        op.get_attr("padding"),
                                        op.get_attr("use_cudnn_on_gpu"),
                                        op.get_attr("data_format"))]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Import data
mnist = input_data.read_data_sets('/tmp/dataset', one_hot=True)

#bez tego pokazuje ze nasz op nie ma zdefiniowanego atrubutu _XlaCompile
#gdy dodamy ten atrybut w C++ to dostajemy wyjatek runtime ze nie moze sparsowac nazwy
jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
with jit_scope():
        # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = weight_variable([5, 5, 1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5,32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 32, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    sess = tf.Session()
    sess.as_default()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    
time_start = time.time()

for i in range(4000):
  batch = mnist.train.next_batch(50)
  if i%10 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    #print(W_conv2.eval(session = sess)[0][0][0])
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print(time.time() - time_start)

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
