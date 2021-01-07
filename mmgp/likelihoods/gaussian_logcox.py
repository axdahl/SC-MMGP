import numpy as np
import tensorflow as tf

from . import likelihood


class GPLogCox(likelihood.Likelihood):
    def __init__(self, offset=0.05, predict_samples=1000):
        '''
        Gaussian likelihood implemented for P=1.
        '''
        self.offset = tf.Variable(offset, dtype=tf.float32)
        self.predict_samples = predict_samples


    def log_cond_prob(self, outputs, latent):
        _log_lambda = (latent + self.offset) # [S,N,1]
        return tf.squeeze(outputs * _log_lambda - tf.exp(_log_lambda) - tf.lgamma(outputs + 1), axis=2) # [S,N]

        # var = self.raw_std_dev ** 2
        # return -0.5 * tf.log(2.0 * np.pi * var) - ((outputs - latent) ** 2) / (2.0 * var)

    def nlpd_cond_prob(self, outputs, latent):
        '''
        for P = 1 individual task-observation p(yip | fk) = p(yi | fk) = log_cond_prob
        '''
        _log_lambda = (latent + self.offset)
        return outputs * _log_lambda - tf.exp(_log_lambda) - tf.lgamma(outputs + 1)

    def get_params(self):
        return [self.offset]

    def predict(self, latent_means, latent_vars):
        """    
        calculate E(Y) and V(Y) for each Y_np = y
        E(y|f) = lambda, V(y|f) = lambda
        lambda_s = exp(f_s + offset) where f_s is sample from posterior
        E(y) ~= mean (lambda_s) 
        Using decomposition of variance
        V(y) = E_{f}(lambda) + V_{f}(lambda)
             ~= mean(lambda_s) + 1/(S-1) * sum((lambda_s - E(y))^2)
        """
        num_points = tf.shape(latent_means)[0]
        lat_dims = tf.shape(latent_means)[1] # Q=1
        latent = (latent_means + tf.sqrt(latent_vars) *
            tf.random_normal([self.predict_samples, num_points, lat_dims])) #[S, N, 1]

        _lambda = tf.exp(latent + self.offset)
        pred_means = tf.reduce_mean(_lambda, 0) # mean over samples [N x 1]
        pred_vars = pred_means + tf.reduce_sum((_lambda - pred_means) ** 2, 0) / (self.predict_samples - 1.0)
        return pred_means, pred_vars

