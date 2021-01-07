import os
import sys

#Compile the TensorFlow ops.
compile_command = ("g++ -std=c++11 -shared ./mmgp/util/tf_ops/vec_to_tri.cc "
                   "./mmgp/util/tf_ops/tri_to_vec.cc -o ./mmgp/util/tf_ops/matpackops.so "
                   "-fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')")

#if sys.platform == "darwin":
#    compile_command += " -undefined dynamic_lookup"


compile_command += " -D_GLIBCXX_USE_CXX11_ABI=0 "
#if sys.platform == "linux2":
#    compile_command += " -D_GLIBCXX_USE_CXX11_ABI=0 "

os.system(compile_command)
