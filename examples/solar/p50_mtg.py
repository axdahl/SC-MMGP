# -*- coding: utf-8 -*-
"""
Script to execute example multi task GP with task-specific features (Bonilla 2008).
The model invokes the IndependentMMGP class.

Inputs: Data training and test sets (dictionary pickle)
Data for example:
 - normalised solar data for 50 sites for 15 minute forecast
 - N_train = 210000, N_test = 113800, P=1, D = 5
 - features = [lag0, lag1, lat, long, t]

Model Options:
 - Diagonal or non diagonal variational posterior covariance Sj (set bool DIAG_POST)
 - Sparse or full posterior covariance (when non diagonal posterior; set bool SPARSE_POST)

Current Settings (sparse non diagonal posterior):
    DIAG_POST = False
    SPARSE_POST = True

"""

import os
import numpy as np
import pickle
import pandas as pd
import traceback
import time
import sklearn.cluster
import csv
import sys

import mmgp
from mmgp import likelihoods
from mmgp import kernels
import tensorflow as tf
from mmgp import datasets
from mmgp import losses
from mmgp  import util

dpath = '/experiments/datasets/'
dfile = 'p50_mtg_inputsdict.pickle'
outdir = '/experiments/results/p50_mtg'

try:
    os.makedirs(outdir)
except FileExistsError:
    pass

def get_inputs():
    """
    inputsdict contains {'Yte': Yte, 'Ytr': Ytr, 'Xtr': Xtr, 'Xte': Xte} where values are np.arrays
    np. arrays are truncated to evenly split into batches of size = batchsize
    """
    with open(os.path.join(dpath, dfile), 'rb') as f:
        d_all = pickle.load(f)

    return d_all


def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


FLAGS = util.util.get_flags()
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
DISPLAY_STEP = FLAGS.display_step
EPOCHS = FLAGS.n_epochs
NUM_SAMPLES =  FLAGS.mc_train
PRED_SAMPLES = FLAGS.mc_test
NUM_INDUCING = 928
NUM_COMPONENTS = FLAGS.num_components
IS_ARD = FLAGS.is_ard
TOL = FLAGS.opt_tol
VAR_STEPS = FLAGS.var_steps
DIAG_POST = False
SPARSE_POST = True # option for non-diag post
MAXTIME = 1200
save_nlpds = False # If True saves samples of nlpds for n,p,s
print("settings done")

# define P and Q
output_dim = 1 #P
node_dim = 1    #Q
lag_dim = 2
startsite = 0 # ignore for mtg

# extract dataset
d = get_inputs()
Ytr, Yte, Xtr, Xte = d['Ytr'], d['Yte'], d['Xtr'], d['Xte']
Ytr, Yte = Ytr.reshape(Ytr.shape[0],1), Yte.reshape(Yte.shape[0],1)

data = datasets.DataSet(Xtr.astype(np.float32), Ytr.astype(np.float32), shuffle=False)
test = datasets.DataSet(Xte.astype(np.float32), Yte.astype(np.float32), shuffle=False)
print("dataset created")

klags = kernels.RadialBasisSlice(lag_dim, active_dims=[0,1], lengthscale=0.01,
                                        std_dev = 0.0, input_scaling = IS_ARD, white = 0.05)
kspat = kernels.CompositeKernel('mul',[kernels.RadialBasisSlice(2, active_dims=[2,3],
                                        std_dev=2.0, lengthscale=0.5, white=0.05, input_scaling = IS_ARD),
                                        kernels.CompactSlice(2, active_dims=[2,3], lengthscale = 2.0,
                                        white = 0.05, input_scaling = IS_ARD)] )

ktime = kernels.PeriodicSliceFixed(1, active_dims=[Xtr.shape[1]-1], lengthscale=0.5,
                                        std_dev=1.5, period=144, white = 0.05)
k_spattemp = kernels.CompositeKernel('mul', kernel_inputs=[kspat, ktime])
kern = [kernels.CompositeKernel('mul', kernel_inputs=[klags, k_spattemp])]
likelihood = likelihoods.Gaussian(0.1)
print("likelihood and kernels set")

Z = init_z(data.X, NUM_INDUCING)
m = mmgp.IndependentMMGP(output_dim, likelihood, kern, Z, num_components=NUM_COMPONENTS,
                                                                        diag_post=DIAG_POST,
                                                                        sparse_post=SPARSE_POST,
                                                                        num_samples=NUM_SAMPLES,
                                                                        predict_samples=PRED_SAMPLES)
print("inducing points and model set")


# initialise losses and logging
error_rate = losses.RootMeanSqError(data.Dout)
os.chdir(outdir)
with open("log_results.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'fit_runtime', 'nelbo', error_rate.get_name(),'generalised_nlpd'])
with open("log_params.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'raw_kernel_params', 'raw_kernlink_params', 'raw_likelihood_params', 'raw_weights'])
with open("log_comp_time.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'batch_time', 'nelbo_time', 'pred_time', 'gen_nlpd_time', error_rate.get_name()+'_time'])


# optimize
o = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9,beta2=0.99)
print("start time = ", time.strftime('%X %x %Z'))
m.fit(data, o,  var_steps = VAR_STEPS, epochs = EPOCHS, batch_size = BATCH_SIZE, display_step=DISPLAY_STEP, test = test,
        loss = error_rate, tolerance = TOL, max_time=MAXTIME )
print("optimisation complete")


# export final predicted values and loss metrics
ypred = m.predict(test.X, batch_size = BATCH_SIZE) #same batchsize used for convenience
np.savetxt("predictions.csv", np.concatenate(ypred, axis=1), delimiter=",")
if save_nlpds == True:
    nlpd_samples, nlpd_meanvar = m.nlpd_samples(test.X, test.Y, batch_size = BATCH_SIZE)
    print('check dimensions of nlpd_samples and nlpd_meanvar: ', nlpd_samples.shape, nlpd_meanvar.shape)
    print('check type of nlpd_samples and nlpd_meanvar: ', nlpd_samples.dtype, nlpd_meanvar.dtype)
    try:
        np.savetxt("nlpd_meanvar.csv", nlpd_meanvar, delimiter=",")
    except:
        print('nlpd_meanvar export fail')
    try:
        np.savetxt("nlpd_samples.csv", nlpd_samples, delimiter=",")  # NP x S (NxS concat for P tasks)
    except:
        print('nlpd_samples export fail')
print("Final " + error_rate.get_name() + "=" + "%.4f" % error_rate.eval(test.Y, ypred[0]))
print("Final " + "generalised_nlpd" + "=" + "%.4f" % m.nlpd_general(test.X, test.Y, batch_size = BATCH_SIZE))


# extra accuracy measures at end of routine
error_rate_end = [losses.MeanAbsError(data.Dout)]
print("Final ", [e.get_name() for e in error_rate_end])
print([e.eval(test.Y, ypred[0]) for e in error_rate_end])
predvar = [np.mean(np.mean(ypred[1]))]
print("Final predvar ", predvar)

with open("final_losses.csv", 'w', newline='') as f:
    csv.writer(f).writerows([[e.get_name() for e in error_rate_end] + ['pred_var'],
                            [e.eval(test.Y, ypred[0]) for e in error_rate_end] + predvar])

print("finish time = " + time.strftime('%X %x %Z'))
