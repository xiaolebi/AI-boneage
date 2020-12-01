from __future__ import print_function,absolute_import
import errno
import os
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def normalizedME(output,target,w,h):
    batch_size = target.size(0)
    diff = output - target
    diff = np.sqrt(diff.T * diff)/(w*h)
    return diff/batch_size

class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count