# -*- coding: utf-8 -*-
"""
Script to execute example covarying MMGP regression forecasting model
with full Krhh.

Inputs: Data training and test sets (dictionary pickle)
Data for example:
 - normalised solar data for 25 sites for 15 minute forecast
 - N_train = 4200, N_test = 2276, P = 25, D = 51
 - Xtr[:, :50] 2 recent lagged observations for each site in order
 - Xtr[:, 50] time index
 - link inputs is a 25x2 array (link inputs repeated for every group)
   with normalised lat,long for each site in order

Model Options:
 - Sparse or full x-function covariance prior Krhh (set bool SPARSE_PRIOR)
 - Diagonal or Kronecker-structured variational posterior covariance Sr (set bool DIAG_POST)
 - Sparse or full posterior covariance (when Kronecker posterior; set bool SPARSE_POST)

Current Settings (sparse covarying mmgp model with sparse Kronecker posterior):
    DIAG_POST = False
    SPARSE_PRIOR = False  # set True for equivalent sparse scmmgp model
    SPARSE_POST = True

Note on specifying group structure for F:
    Grouping occurs via block_struct, a nested list of grouping order
    Where functions [i] are independent i.e. in own block, set link_kernel[i] = link_inputs[i] = 1.0
    See model class preamble and example below for further details.

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
dfile = 'p25_inputsdict.pickle'
dlinkfile = 'p25_linkinputsarray.pickle'
outdir = '/experiments/results/p25_nonsparse_cmmgp/'

try:
    os.makedirs(outdir)
except FileExistsError:
    pass

def get_inputs():
    """
    inputsdict contains {'Yte': Yte, 'Ytr': Ytr, 'Xtr': Xtr, 'Xte': Xte} where values are np.arrays
    np. arrays are truncated to evenly split into batches of size = batchsize

    returns inputsdict, Xtr_link (ndarray, shape = [P, D_link_features])
    """
    with open(os.path.join(dpath, dfile), 'rb') as f:
        d_all = pickle.load(f)

    with open(os.path.join(dpath, dlinkfile), 'rb') as f:
        d_link = pickle.load(f)

    return d_all, d_link


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
NUM_INDUCING = FLAGS.n_inducing
NUM_COMPONENTS = FLAGS.num_components
IS_ARD = FLAGS.is_ard
TOL = FLAGS.opt_tol
VAR_STEPS = FLAGS.var_steps
DIAG_POST = False
SPARSE_PRIOR = False
SPARSE_POST = True # option for non-diag post
MAXTIME = 1200
print("settings done")


# define GPRN P and Q
output_dim = 25 #P
node_dim = 25    #Q
lag_dim = 2
save_nlpds = False # If True saves samples of nlpds for n,p,s

# extract dataset
d, d_link = get_inputs()
Ytr, Yte, Xtr, Xte = d['Ytr'], d['Yte'], d['Xtr'], d['Xte']

data = datasets.DataSet(Xtr.astype(np.float32), Ytr.astype(np.float32), shuffle=False)
test = datasets.DataSet(Xte.astype(np.float32), Yte.astype(np.float32), shuffle=False)
print("dataset created")


# model config block rows (where P=Q): block all w.1, w.2 etc, leave f independent
# order of block_struct is rows, node functions
# lists required: block_struct, link_inputs, kern_link, kern

#block_struct nested list of grouping order
weight_struct = [[] for _ in range(output_dim)]
for i in range(output_dim):
    row = list(range(i, i+output_dim*(node_dim-1)+1, output_dim))
    row_0 = row.pop(i)  # bring diag to pivot position
    weight_struct[i] = [row_0] + row


nodes = [[x] for x in list(range(output_dim * node_dim, output_dim * node_dim + output_dim))]
block_struct = weight_struct + nodes


# create link inputs (link inputs used repeatedly but can have link input per group)
# permute to bring diagonal to first position
link_inputs = [[] for _ in range(output_dim)]
for i in range(output_dim):
    idx = list(range(d_link.shape[0]))
    link_inputs[i] = d_link[[idx.pop(i)] + idx, :]

link_inputs = link_inputs + [1.0 for i in range(output_dim)]  # for full W row blocks, independent nodes

# create 'between' kernel list
klink_rows = [kernels.CompositeKernel('mul',[kernels.RadialBasis(2, std_dev=2.0, lengthscale=1.0, white=0.01, input_scaling = IS_ARD),
                                            kernels.CompactSlice(2, active_dims=[0,1], lengthscale = 2.0, input_scaling = IS_ARD)] )
                                            for i in range(output_dim) ]
klink_f = [1.0 for i in range(node_dim)]

kernlink = klink_rows +  klink_f


# create 'within' kernel
# kern
lag_active_dims_s = [ [] for _ in range(output_dim)]
for i in range(output_dim):
    lag_active_dims_s[i] = list(range(lag_dim*i, lag_dim*(i+1)))

k_rows = [kernels.CompositeKernel('mul',[kernels.RadialBasisSlice(lag_dim, active_dims=lag_active_dims_s[i],
                                            std_dev = 1.0, white = 0.01, input_scaling = IS_ARD),
                                            kernels.PeriodicSliceFixed(1, active_dims=[Xtr.shape[1]-1],
                                            lengthscale=0.5, std_dev=1.0, period = 144) ])
                                            for i in range(output_dim)]
k_f = [kernels.RadialBasisSlice(lag_dim, active_dims=lag_active_dims_s[i], std_dev = 1.0, white = 0.01, input_scaling = IS_ARD)
    for i in range(output_dim)]

kern = k_rows + k_f

print('len link_inputs ',len(link_inputs))
print('len kernlink ',len(kernlink))
print('len kern ', len(kern))
print('no. groups = ', len(block_struct), 'no. latent functions =', len([i for b in block_struct for i in b]))
print('number latent functions', node_dim*(output_dim+1))


likelihood = likelihoods.CovaryingRegressionNetwork(output_dim, node_dim, std_dev = 0.2)  # p, q, lik_noise
print("likelihood and kernels set")
Z = init_z(data.X, NUM_INDUCING)
print('inducing points set')
m = mmgp.ExplicitSCMMGP(output_dim, likelihood, kern, kernlink, block_struct, Z, link_inputs,
    num_components=NUM_COMPONENTS, diag_post=DIAG_POST, sparse_prior=SPARSE_PRIOR,
    sparse_post=SPARSE_POST, num_samples=NUM_SAMPLES, predict_samples=PRED_SAMPLES)
print("model set")


# initialise losses and logging
error_rate = losses.RootMeanSqError(data.Dout)

os.chdir(outdir)
with open("log_results.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'fit_runtime', 'nelbo', error_rate.get_name(),'generalised_nlpd'])
with open("log_params.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'raw_kernel_params', 'raw_kernlink_params', 'raw_likelihood_params', 'raw_weights'])
with open("log_comp_time.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'batch_time', 'nelbo_time', 'pred_time', 'gen_nlpd_time', error_rate.get_name()+'_time'])

# optimise
o = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9,beta2=0.99)
print("start time = ", time.strftime('%X %x %Z'))
m.fit(data, o,  var_steps = VAR_STEPS, epochs = EPOCHS, batch_size = BATCH_SIZE, display_step=DISPLAY_STEP,
        test = test, loss = error_rate, tolerance = TOL, max_time=MAXTIME )
print("optimisation complete")



# export final predicted values and loss metrics
ypred = m.predict(test.X, batch_size = BATCH_SIZE) #same batchsize used for convenience
np.savetxt("predictions.csv", np.concatenate(ypred, axis=1), delimiter=",")

if save_nlpds == True:
    nlpd_samples, nlpd_meanvar = m.nlpd_samples(test.X, test.Y, batch_size = BATCH_SIZE)
    try:
        np.savetxt("nlpd_meanvar.csv", nlpd_meanvar, delimiter=",")  # N x 2P as for predictions
    except:
        print('nlpd_meanvar export fail')
    try:
        np.savetxt("nlpd_samples.csv", nlpd_samples, delimiter=",")  # NP x S (NxS concat for P tasks)
    except:
        print('nlpd_samples export fail')

print("Final " + error_rate.get_name() + "=" + "%.4f" % error_rate.eval(test.Y, ypred[0]))
print("Final " + "generalised_nlpd" + "=" + "%.4f" % m.nlpd_general(test.X, test.Y, batch_size = BATCH_SIZE))
error_rate_end = [losses.MeanAbsError(data.Dout)] # any extra accuracy measures at end of routine
print("Final ", [e.get_name() for e in error_rate_end])
print([e.eval(test.Y, ypred[0]) for e in error_rate_end])
predvar = [np.mean(np.mean(ypred[1]))]
print("Final predvar ", predvar)

with open("final_losses.csv", 'w', newline='') as f:
    csv.writer(f).writerows([[e.get_name() for e in error_rate_end] + ['pred_var'],
                            [e.eval(test.Y, ypred[0]) for e in error_rate_end] + predvar])

print("finish time = " + time.strftime('%X %x %Z'))
