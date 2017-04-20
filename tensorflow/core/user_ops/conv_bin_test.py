import os.path

import tensorflow as tf

class ConvBinOpTest(tf.test.TestCase):

  def testBasic(self):
    library_filename = os.path.join(tf.resource_loader.get_data_files_path(),
                                    'conv_bin.so')
    conv_bin = tf.load_op_library(library_filename)

if __name__ == '__main__':
  tf.test.main()
