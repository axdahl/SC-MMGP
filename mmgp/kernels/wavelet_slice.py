'''
Wavelet kernel
slice allows kernel operation on feature subset
active_dims is iterable of feature dimensions to extract
input_dim must equal dimension defined by active_dims
'''

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel
from .kernel_extras import *


class WaveletSlice(kernel.Kernel):

    def __init__(self, input_dim, active_dims=None, shift=0, scale = 0.01,
                 white=0.01, input_scaling=False):
        if input_scaling:
            self.shift = tf.Variable(shift * tf.ones([input_dim]))
            self.scale = tf.Variable(scale * tf.ones([input_dim]))
        else:
            self.shift = tf.Variable([shift], dtype=tf.float32)
            self.scale = tf.Variable([scale], dtype=tf.float32)
        self.input_dim = input_dim
        self.active_dims = active_dims
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = (self.white * util.eye(tf.shape(points1)[0]) +
                0.1 * self.white * tf.ones( [tf.shape(points1)[0], tf.shape(points1)[0]]))
        else:
            white_noise = 0.01 * self.white * tf.ones( [tf.shape(points1)[0], tf.shape(points2)[0]] )

        points1, points2 = dim_slice(self, points1, points2)

        def h(x):
            # Zhang wavelet
            #return tf.cos(1.75*x)*tf.exp(-0.5*x**2)
            # mexican hat wavelet
            return (1-x**2)*tf.exp(-0.5*x**2)

        kern1, kern2 = h((points1 - self.shift)/tf.exp(self.scale)), h((points2 - self.shift)/tf.exp(self.scale))
        kern1, kern2 = tf.reduce_prod(kern1, axis=1), tf.reduce_prod(kern2, axis=1)
        kern = tf.einsum('i,j->ij', kern1, kern2)
        return kern + white_noise

    def diag_kernel(self, points):
        def h(x):
            # Zhang wavelet
            return tf.cos(1.75*x)*tf.exp(-0.5*x**2)
            # mexican hat wavelet
            #return (1-x**2)*tf.exp(-0.5*x**2)

        points = dim_slice_diag(self, points)
        kern = tf.reduce_prod(h((points - self.shift)/tf.exp(self.scale)) , axis=1) **2
        return kern + self.white

    def get_params(self):
        return [self.shift, self.scale]
