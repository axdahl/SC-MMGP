import copy

import tensorflow as tf

def init_list(init, dims): #list of dims creates nested lists
    def empty_list(dims):
        if not dims:
            return None
        else:
            return [copy.deepcopy(empty_list(dims[1:])) for i in range(dims[0])]

    def fill_list(dims, l):
        if len(dims) == 1:
            for i in range(dims[0]):
                if callable(init):
                    l[i] = init()
                else:
                    l[i] = init
        else:
            for i in range(dims[0]):
                fill_list(dims[1:], l[i])

    l = empty_list(dims) 
    fill_list(dims, l)

    return l


def ceil_divide(dividend, divisor):
    return (dividend + divisor - 1) / divisor


def log_cholesky_det(chol):
    return 2 * tf.reduce_sum(tf.log(tf.diag_part(chol)))


def diag_mul(mat1, mat2):
    return tf.reduce_sum(mat1 * tf.transpose(mat2), 1)


def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m), dim))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), dim))

def mat_square(mat):
    return tf.matmul(mat, tf.transpose(mat))

def sparse_chol_vec(kdiag, kcol):
    """
    Construct sparse chol from first col and diag of kernel_mat
    Inputs:
        kdiag - vec len==K
        kcol  - vec len==K-1
    """
    MINVAL, MAXVAL = 1e-6, 1e6
    # compute cholesky pivot, first col and diagonal
    kdiag, kcol = tf.squeeze(kdiag), tf.squeeze(kcol) #ensure rank match
    c0 = tf.sqrt(tf.maximum(kdiag[0], MINVAL))
    ccol = kcol / c0
    cdiag = tf.sqrt(tf.maximum(kdiag[1:] - ccol**2, MINVAL))
    # combine pivot row (padded), column and diag
    chol = tf.concat([tf.pad(tf.reshape(c0, [1,1]), [[0,0], [0, tf.shape(ccol)[0]]]),
            tf.concat([tf.expand_dims(ccol,1), tf.diag(cdiag)], axis=1)], axis=0)

    return chol

def sparse_chol_to_inv(chol):
    """
    input sparse chol(K) and return 1st col and diag of precision(K)
    """
    k0 = chol[0,0]**2  # k11
    pii = 1 / tf.diag_part(chol)[1:]**2 # kii.1^-1  Qr-1 x 1 (rank=1)
    pi1 = - pii * (chol[1:,0] / chol[0,0]) # rank=1
    p0 = tf.reshape((1/k0) * (1 + tf.reduce_sum(chol[1:,0] ** 2 * pii)), [1])
    pcol = tf.concat([p0, pi1], axis=0)
    pdiag = tf.concat([p0, pii], axis=0) # Qr x Qr
    return pcol, pdiag

def kronecker_mul(mat1, mat2):
    """
    Computes the Kronecker product two matrices.
    Adapted from:
    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/contrib/kfac/python/ops/utils.py 
    Input matrices must be placeholders == 1.0 (rank=0), or matrices of rank==2 i.e. N x M
    """
    if mat1 == 1.0 or mat2 == 1.0:
        return mat1 * mat2
    else:
        #mat1, mat2 = tf.squeeze(mat1), tf.squeeze(mat2)
        m1, n1 = tf.shape(mat1)[0], tf.shape(mat1)[1]
        mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
        m2, n2 = tf.shape(mat2)[0], tf.shape(mat2)[1]
        mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
        return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])

def block_diag(tensorlist):
    '''
    Block diagonal matrix or tensor constructed
    from list of tensors stacked along diagonal in order.

    Block diag constructed by padding columns of each
    tensor and concatenating along axis=0.

    First two dimensions (rows, cols) can differ, remaining
    dimensions must be equal for all input tensors.
    '''
    padded = []
    for t in tensorlist:
        #if tf.rank(t) != 3:
        #    quit()
        t_i = tensorlist.index(t)
        pre = 0 + sum([tf.shape(i)[1] for i in tensorlist if tensorlist.index(i) < t_i])
        post = 0 + sum([tf.shape(i)[1] for i in tensorlist if tensorlist.index(i) > t_i])
        padding = [ [0,0], [pre,post], [0,0] ] # padding only in dim=1
        padded.append(tf.pad(t, padding, constant_values = 0.0))
        out = tf.concat(padded, axis = 0)
    return out


def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                           'Must divide evenly into the dataset sizes.')
    flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
    flags.DEFINE_integer('n_epochs', 200, 'Number of passes through the data')
    flags.DEFINE_integer('n_inducing', 200, 'Number of inducing points')
    flags.DEFINE_integer('display_step', 5, 'Display progress every FLAGS.display_step iterations')
    flags.DEFINE_integer('mc_train', 200, 'Number of Monte Carlo samples used to compute stochastic gradients')
    flags.DEFINE_integer('mc_test', 500, 'Number of Monte Carlo samples for predictions')
    flags.DEFINE_string('optimizer', "adam", 'Optimizer')
    flags.DEFINE_boolean('is_ard', True, 'Using ARD kernel or isotropic')
    flags.DEFINE_float('lengthscale', 1.0, 'Initial lengthscale')
    flags.DEFINE_integer('var_steps', 1, 'Number of times spent optimizing the variational objective.')
    flags.DEFINE_integer('loocv_steps', 0, 'Number of times spent optimizing the LOOCV objective.')
    flags.DEFINE_float('opt_growth', 0.0, 'Percentage to grow the number of each optimizations.')
    flags.DEFINE_float('opt_tol', 0.00001, 'Proportionate (absolute) change objective threshold')
    flags.DEFINE_integer('num_components', 1, 'Number of mixture components on posterior')
    flags.DEFINE_string('kernel', 'rbf', 'kernel')
    flags.DEFINE_string('device_name', 'cpu0', 'Device name')
    flags.DEFINE_integer('kernel_degree', 0, 'Degree of arccosine kernel')
    flags.DEFINE_integer('kernel_depth', 1, 'Depth of arcosine kernel')
    return FLAGS

