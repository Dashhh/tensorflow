import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 9])
x_reshaped = tf.reshape(x, [-1,3,3,1])
w = np.zeros([2,2,1,1])
w[0:2,0:2,0,0] = np.vstack(([1,1],[1,1]))
W_conv1 = tf.Variable(tf.cast(w,dtype=tf.float32))

conv_bin_module = tf.load_op_library('../tensorflow/core/user_ops/conv_bin_inputs.so')

h_conv_bin = conv_bin_module.binary_conv_input2d(x_reshaped, W_conv1, strides=[1,1,1,1], padding='VALID')

sess = tf.Session()
sess.as_default()
sess.run(tf.global_variables_initializer())

numbers = np.reshape(np.array([1,2,3,4,5,6,7,8,9], dtype=np.float32),(1,9))
result_bin = h_conv_bin.eval(session=sess, feed_dict={x: numbers})
print(result_bin[0,0:2,0:2,0])
