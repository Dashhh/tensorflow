load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

tf_custom_op_library(
    name = "conv_bin.so",
    srcs = ["conv_bin.cc"],
)

tf_py_test(
    name = "conv_bin_test",
    size = "small",
    srcs = ["conv_bin_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    data = [":conv_bin_op.so"],
)
