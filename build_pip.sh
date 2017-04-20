sudo bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
sudo bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip3.5 install /tmp/tensorflow_pkg/tensorflow-1.1.0rc0-cp35-cp35m-linux_x86_64.whl
