#sudo bazel build --config opt //tensorflow/core/user_ops:conv_bin.so
TF_INC=$(python3.5 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -Wall -g -shared conv_bin.cc -o conv_bin.so -fPIC -I $TF_INC -O2 -fopenmp
g++ -std=c++11 -Wall -g -shared conv_bin_inputs.cc -o conv_bin_inputs.so -fPIC -I $TF_INC -O2 -fopenmp
