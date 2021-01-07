import numpy as np
import tensorflow as tf

from . import likelihood
from .. import util

class SCMMGPLogCox(likelihood.Likelihood):
    """
    Implementation of a Log Gaussian Cox process
    p(y|f) = (lambda)^y exp(-lambda) / y!
    lambda = f + offset

    This class takes multiple latent function inputs (assumed covarying GPs)
    that may be linearly combined and pushed through the Poisson transform
    Assumes weight matrix is balanced not ragged i.e. F=WG
    """
    def __init__(self, output_dim, node_dim, offset=0.05): 
        # might need to take in block structure as argument
        self.latfunc_dim = node_dim*(output_dim+1)  # dim Q
        self.output_dim = output_dim # dim P
        self.node_dim = node_dim # Qg
        #TODO check if needs likelihood noise
        self.offset = tf.Variable(np.ones([self.output_dim]) * offset, dtype=tf.float32)       # separate offset for each output
        #self.offset = tf.Variable(offset, dtype=tf.float32) # log-cox shared offset for all latfuncs

    def log_cond_prob(self, outputs, latent):
        #weight_dim = self.latfunc_dim - self.node_dim
        weight_dim = self.output_dim * self.node_dim
        weights = latent[:, :, :weight_dim] #weights up to Q-Qg [:,:,(Q-Qg)]
        inputs = latent[:, :, weight_dim:]     # inputs Qg to end [:,:,Qg]

        # produces S x N x P tensor of predictions (currently assumed balanced W)
        split_weights = tf.stack(tf.split(weights, self.node_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.node_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0) # calculate WG [S, N, P]

        # push through poisson likelihood and sum indiv probs over output dim [S x N]
        _log_lambda = (prod + self.offset)
        return tf.reduce_sum(outputs * _log_lambda - tf.exp(_log_lambda) - tf.lgamma(outputs + 1), 2)

    def nlpd_cond_prob(self, outputs, latent):
        # returns individual log probabilities for each sample for n,p i.e. [S, N, P]
        pq = self.output_dim * self.node_dim
        weights = latent[:, :, :pq] #weights up to p*q [:,:,q*p]
        inputs = latent[:, :, pq:]     # inputs p*q to end [:,:,q]
        split_weights = tf.stack(tf.split(weights, self.node_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.node_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0) # calculate WG [S, N, P]

        _log_lambda = (prod + self.offset)
        return outputs * _log_lambda - tf.exp(_log_lambda) - tf.lgamma(outputs + 1)

    def get_params(self):
        return [self.offset]

    def predict(self, latent):    
        """    
        calculate E(Y) and V(Y) for each Y_np = y
        E(y|f) = lambda, V(y|f) = lambda
        lambda_s = exp(f_s + offset) where f_s is sample of w_i * G
        E(y) ~= mean (lambda_s) 
        Using decomposition of variance
        V(y) = E_{f}(lambda) + V_{f}(lambda)
             ~= mean(lambda_s) + 1/(S-1) * sum((lambda_s - E(y))^2)
        """
        predict_samples_denom = tf.to_float(tf.shape(latent)[0]) - 1.0
        pq = self.output_dim * self.node_dim
        weights = latent[:, :, :pq] #weights up to p*q [:,:,q*p]
        inputs = latent[:, :, pq:]     # inputs p*q to end [:,:,q]
        split_weights = tf.stack(tf.split(weights, self.node_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.node_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0) # calculate WG [S, N, P]
        
        _lambda = tf.exp(prod + self.offset)
        pred_means = tf.reduce_mean(_lambda, 0) # mean over samples [N x P]
        pred_vars = pred_means + tf.reduce_sum((_lambda - pred_means) ** 2, 0) / predict_samples_denom
        return pred_means, pred_vars
