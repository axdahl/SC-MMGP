"""
Script to execute example explicit sparse covarying MMGP with Poisson likelihood.
The model invokes the 'explicit' sparse model class and accepts a non-degenerate kernel.

Inputs: Data training and test sets (dictionary pickle)
Data for example:
 - count data for 50 airports
 - N_train = 20,000, N_test = 10,798, P = 50, D = 105
 - Xtr[:, :4] ['time_index', 'dayofweek', 'dayofmonth', 'month']
 - Xtr[:, 4:105] total scheduled arrivals and departures per airport
 - Xtr[:, 105] total activity (arrivals and departures) for all airports
 - link inputs is a 50x3 array (link inputs repeated for every group)
   with normalised lat,long and airport size (total scheduled flights over sample period)

Model Options:
 - Sparse or full x-function covariance prior Krhh (set bool SPARSE_PRIOR)
 - Diagonal or Kronecker-structured variational posterior covariance Sr (set bool DIAG_POST)
 - Sparse or full posterior covariance (when Kronecker posterior; set bool SPARSE_POST)

Current Settings (general scmmgp model with sparse Kronecker posterior):
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
dfile = 'logcox_nozeroy_aggx_inputsdict.pickle'
dlinkfile = 'logcox_nozeroy_aggx_linkinputsarray.pickle'
outdir = '/experiments/results/logcox_nonsparse_cmmgp'
siteinclude = os.path.join(dpath, "airports_top50.csv")  # contains order of output variables

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
BATCH_SIZE = 100
LEARNING_RATE = FLAGS.learning_rate
DISPLAY_STEP = FLAGS.display_step
EPOCHS = 150
NUM_SAMPLES = 200
PRED_SAMPLES = 500
NUM_INDUCING = 250
NUM_COMPONENTS = FLAGS.num_components
IS_ARD = FLAGS.is_ard
TOL = 0.0001
VAR_STEPS = FLAGS.var_steps
DIAG_POST = False
SPARSE_PRIOR = False
SPARSE_POST = True # option for non-diag post
MAXTIME = 1200
save_nlpds = True # If True saves samples of nlpds (mean and variance)

print("settings done")


# define GPRN P and Q
output_dim = 50 #P
locfeat_dim = 2 # [scheduled arrivals, scheduled departures] for time increment for airport
commonfeats = list(range(4)) # [t_ix, dayofweek, dayofmonth, month]
num_hubs = 5 # becomes nodedim
toplist = ['ATL',
    'ORD',
    'DFW',
    'DEN',
    'LAX',
    'PHX',
    'IAH',
    'LAS',
    'DTW',
    'EWR']

use_sites = pd.read_csv(siteinclude,header=None).iloc[:,0].tolist() # order of output variables
toplist = toplist[:num_hubs]
hublocs = [use_sites.index(x) for x in toplist]
nonhubs = [use_sites.index(x) for x in use_sites if x not in toplist] #non hub dims
print(toplist)
print(hublocs)
print(nonhubs)
node_dim = len(hublocs)    #Q

# extract dataset
d, d_link = get_inputs()
Ytr, Yte, Xtr, Xte = d['Ytr'], d['Yte'], d['Xtr'], d['Xte']

data = datasets.DataSet(Xtr.astype(np.float32), Ytr.astype(np.float32), shuffle=False)
test = datasets.DataSet(Xte.astype(np.float32), Yte.astype(np.float32), shuffle=False)
print("dataset created")

# lists required: block_struct, link_inputs, kern_link, kern
# model config: block columns, leave f independent
# order of block_struct is columns, node functions

#block_struct nested list of grouping order
weight_struct = [[] for _ in range(node_dim)]

for i in range(node_dim):
    col = list(range(i*output_dim, i*output_dim + output_dim))
    col_0 = col.pop(hublocs[i])  # bring hub to pivot position
    weight_struct[i] = [col_0] + col

nodes = [[x] for x in list(range(output_dim * node_dim, output_dim * node_dim + node_dim))]
block_struct = weight_struct + nodes

# create link inputs (link inputs used repeatedly but can have link input per group)
# permute to bring hub to first position
link_inputs = [[] for _ in range(node_dim)]
for i in range(node_dim):
    idx = list(range(d_link.shape[0]))
    link_inputs[i] = d_link[[idx.pop(hublocs[i])] + idx, :]  # match inputs order to block_struct

link_inputs = link_inputs + [1.0 for i in range(node_dim)]  # for full W column blocks, independent nodes

# link kernel
klink_w = [kernels.RadialBasisSlice(3, active_dims=[0,1,2], std_dev=2.0, lengthscale=1.0, white=0.01,
                                             input_scaling = IS_ARD)
                                             for i in range(len(weight_struct)) ]
klink_f = [1.0 for i in range(node_dim)]
kernlink = klink_w +  klink_f

# create 'within' kernel
# kern
k_w = [kernels.CompositeKernel('add',[kernels.RadialBasisSlice(Xtr.shape[1],
                                            active_dims= list(range(Xtr.shape[1])),
                                            std_dev = 1.0, white = 0.01, input_scaling = IS_ARD),
                                            kernels.PeriodicSlice(1, active_dims=[0],
                                            lengthscale=0.5, std_dev=1.0, period = 2.0) ])
                                            for i in range(len(weight_struct))]
k_f = [kernels.RadialBasisSlice(1, active_dims=[0], std_dev = 1.0, white = 0.01, input_scaling = IS_ARD)
    for i in range(node_dim)]

kern = k_w + k_f

print('len link_inputs ',len(link_inputs))
print('len kernlink ',len(kernlink))
print('len kern ', len(kern))
print('no. groups = ', len(block_struct), 'no. latent functions =', len([i for b in block_struct for i in b]))
print('number latent functions', node_dim*(output_dim+1))

likelihood = likelihoods.SCMMGPLogCox(output_dim, node_dim, offset = 0.05)  # output_dim, node_dim, offset
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
    #try:
    #    np.savetxt("nlpd_samples.csv", nlpd_samples, delimiter=",")  # NP x S (NxS concat for P tasks)
    #except:
    #    print('nlpd_samples export fail')

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
