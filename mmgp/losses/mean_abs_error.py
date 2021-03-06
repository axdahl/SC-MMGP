'''
MAE
'''

import numpy as np
from . import loss

class MeanAbsError(loss.Loss):
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
        error_rate = np.mean(np.absolute(ytrue - ypred))
        return error_rate

    def get_name(self):
        return "MAE"
