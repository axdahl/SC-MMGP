import numpy as np
import tensorflow as tf

from . import likelihood
from .. import util


class CoregionLogCox(likelihood.Likelihood):
    def __init__(self, output_dim, latfunc_dim, weight, offset=0.05, predict_samples=1000):
        self.latfunc_dim = latfunc_dim  # dim q (Qg)
        self.output_dim = output_dim # dim p
        self.predict_samples = predict_samples
        self.weights = tf.Variable(np.ones([self.output_dim, self.latfunc_dim]) * weight, dtype = tf.float32)
        self.offset = tf.Variable(np.ones([self.output_dim]) * offset,
                            dtype=tf.float32)       # separate offset for each output
        #self.offset = tf.Variable(offset, dtype=tf.float32) # log-cox shared offset for all latfuncs
        
    def log_cond_prob(self, outputs, latent):
        # propogate weights [p,q] to split_weights [q,S,n,p]
        weights = tf.transpose(self.weights) #now [q,p]
        split_weights = tf.tile(tf.expand_dims(weights, axis = 1), multiples = [1,tf.shape(latent)[0],1])
        split_weights = tf.tile(tf.expand_dims(split_weights, axis = 2), multiples = [1,1,tf.shape(latent)[1],1])

        split_inputs = tf.stack(tf.split(latent, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0) # calculate WG [S, N, P]

        # push through poisson likelihood and sum indiv probs over output dim [S x N]
        _log_lambda = (prod + self.offset)
        return tf.reduce_sum(outputs * _log_lambda - tf.exp(_log_lambda) - tf.lgamma(outputs + 1), 2)

    def nlpd_cond_prob(self, outputs, latent):
        # returns individual log probabilities for each sample for n,p i.e. [S, N, P]
        weights = tf.transpose(self.weights) #now [q,p]
        split_weights = tf.tile(tf.expand_dims(weights, axis = 1), multiples = [1,tf.shape(latent)[0],1])
        split_weights = tf.tile(tf.expand_dims(split_weights, axis = 2), multiples = [1,1,tf.shape(latent)[1],1])

        split_inputs = tf.stack(tf.split(latent, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)  # calculate WG [S, N, P]

        _log_lambda = (prod + self.offset)
        return outputs * _log_lambda - tf.exp(_log_lambda) - tf.lgamma(outputs + 1)

    def get_params(self):
        return [self.weights, self.offset]

    def predict(self, latent_means, latent_vars):
        """    
        calculate E(Y) and V(Y) for each Y_np = y
        E(y|f) = lambda, V(y|f) = lambda
        lambda_s = exp(f_s + offset) where f_s is sample of w_i * G
        E(y) ~= mean (lambda_s) 
        Using decomposition of variance
        V(y) = E_{f}(lambda) + V_{f}(lambda)
             ~= mean(lambda_s) + 1/(S-1) * sum((lambda_s - E(y))^2)
        """
        num_points = tf.shape(latent_means)[0]
        lat_dims = tf.shape(latent_means)[1] 
        latent = (latent_means + tf.sqrt(latent_vars) *
                tf.random_normal([self.predict_samples, num_points, lat_dims])) #[S, N, Q]

        # propogate weights [p,q] to split_weights [q,S,n,p]
        weights = tf.transpose(self.weights) #now [q,p]
        split_weights = tf.tile(tf.expand_dims(weights, axis = 1), multiples = [1,tf.shape(latent)[0],1])
        split_weights = tf.tile(tf.expand_dims(split_weights, axis = 2), multiples = [1,1,tf.shape(latent)[1],1])

        split_inputs = tf.stack(tf.split(latent, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0) # calculate WG [S, N, P]

        _lambda = tf.exp(prod + self.offset)
        pred_means = tf.reduce_mean(_lambda, 0) # mean over samples [N x P]
        pred_vars = pred_means + tf.reduce_sum((_lambda - pred_means) ** 2, 0) / (self.predict_samples - 1.0)
        return pred_means, pred_vars

