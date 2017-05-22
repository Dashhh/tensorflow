from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

def conv_bin_cpp(x, W,padding):
    conv_bin_module = tf.load_op_library('../tensorflow/core/user_ops/conv_bin.so')
    return conv_bin_module.binary_conv2d(x, W, strides=[1,1,1,1], padding=padding)


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
  h_conv_bin = conv_bin_cpp(x_reshaped, W_conv1, padding=padding)
  sess = tf.Session()
  sess.as_default()
  sess.run(tf.global_variables_initializer())
  result_bin = h_conv_bin.eval(session=sess, feed_dict={x: input})
  return result_bin

def compute_grad_python(input_shape, x_shape,w, input, padding):
  x = tf.placeholder(tf.float32, input_shape)
  x_reshaped = tf.reshape(x, x_shape)
  W_conv1 = tf.Variable(tf.cast(w,dtype=tf.float32))
  h_conv_bin = conv_bin(x_reshaped, W_conv1, padding)
  grad = tf.gradients(h_conv_bin, W_conv1)  
  sess = tf.Session()
  sess.as_default()
  sess.run(tf.global_variables_initializer())
  grad_evaled = grad[0].eval(session=sess, feed_dict={x: input})
  return grad_evaled 

def compute_grad_cpp(input_shape, x_shape,w, input, padding):
  jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
  with jit_scope():
    x = tf.placeholder(tf.float32, input_shape)
    x_reshaped = tf.reshape(x, x_shape)
    W_conv1 = tf.Variable(tf.cast(w,dtype=tf.float32))
    h_conv_bin = conv_bin_cpp(x_reshaped, W_conv1, padding=padding)
    grad = tf.gradients(h_conv_bin, W_conv1)  
    sess = tf.Session()
    sess.as_default()
    sess.run(tf.global_variables_initializer())
    grad_evaled = grad[0].eval(session=sess, feed_dict={x: input})
    return grad_evaled 

class TestStringMethods(unittest.TestCase):
  def test_grad_simple(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,1],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    cpp = compute_grad_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_grad_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

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

  def test_negative_weights(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-2],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    cpp = compute_ff_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

  def test_negative_weights_multibatch(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-2],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9,1,2,3,20,11,33,-5,1,9], dtype=np.float32),(2,9))
    cpp = compute_ff_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

  def test_grad_negative_weights_when_same_padding(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-2],[1,1]))
    padding = 'SAME'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9,1,2,3,20,11,33,-5,1,9], dtype=np.float32),(2,9))
    cpp = compute_grad_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_grad_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))

  def test_grad_negative_weights(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-1.1],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,1,1,1,1,1,1,1,1], dtype=np.float32),(1,9))
    cpp = compute_grad_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_grad_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))
  
  def test_grad_negative_weights_multibatch(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-2],[1,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9,1,2,3,20,11,33,-5,1,9], dtype=np.float32),(2,9))
    cpp = compute_grad_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_grad_python(input_shape,x_shape,w,input, padding)
    self.assertTrue(np.array_equal(cpp, python))
    
  def test_floating_weights(self):
    x_shape = [-1,3,3,1]
    input_shape = [None,9]
    w = np.zeros([2,2,1,1])
    w[0:2,0:2,0,0] = np.vstack(([1,-0.5],[0.5,1]))
    padding = 'VALID'
    input = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
    cpp = compute_ff_cpp(input_shape, x_shape, w, input, padding)   
    python = compute_ff_python(input_shape,x_shape,w,input, padding)    
    self.assertTrue(np.array_equal(cpp, python))
    
if __name__ == '__main__':
  unittest.main()
