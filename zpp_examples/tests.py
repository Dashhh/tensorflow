from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import unittest
import tensorflow as tf
import numpy as np

def conv2d(x, W, padding):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def conv_bin(x, W, padding):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
                w_shape = tf.shape(W)
                n = tf.cast(tf.reduce_prod(w_shape[0:-1]),tf.float32) 
                abs = tf.abs(W)
                a = tf.stop_gradient(tf.reduce_sum(abs, [0,1,2])/n)  
                w_sign = tf.sign(W/a)              
                return conv2d(x, w_sign, padding)*a

def compute_ff_python(input_shape, x_shape, w, input, padding):
  x = tf.placeholder(tf.float32, input_shape)
  x_reshaped = tf.reshape(x, x_shape)
  W_conv1 = tf.Variable(tf.cast(w,dtype=tf.float32))
  h_conv_bin = conv_bin(x_reshaped, W_conv1, padding)
  sess = tf.Session()
  sess.as_default()
  sess.run(tf.global_variables_initializer())
  result_bin = h_conv_bin.eval(session=sess, feed_dict={x: input})
  return result_bin

def compute_ff_cpp(input_shape, x_shape, w, input, padding):
  x = tf.placeholder(tf.float32, input_shape)
  x_reshaped = tf.reshape(x, x_shape)
  W_conv1 = tf.Variable(tf.cast(w,dtype=tf.float32))
  conv_bin_module = tf.load_op_library('../tensorflow/core/user_ops/conv_bin.so')
  h_conv_bin = conv_bin_module.binary_conv2d(x_reshaped, W_conv1, strides=[1,1,1,1], padding=padding)
  sess = tf.Session()
  sess.as_default()
  sess.run(tf.global_variables_initializer())
  result_bin = h_conv_bin.eval(session=sess, feed_dict={x: input})
  return result_bin

class TestStringMethods(unittest.TestCase):
  def test_simple(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,1],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    cpp = compute_ff_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

  def test_same_padding(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,1],[1,1]))
    padding = 'SAME'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    cpp = compute_ff_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

  def test_multiple_inputs(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,1],[1,1]))
    numbers = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    input = np.vstack((numbers,numbers))    
    padding = 'VALID'
    cpp = compute_ff_cpp(input_shape, x_shape, w, input,padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

  def test_negative_weights(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-2],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    cpp = compute_ff_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    print('python',python)
    print('cpp', cpp)
    self.assertTrue(np.array_equal(cpp, python))

if __name__ == '__main__':
  unittest.main()
