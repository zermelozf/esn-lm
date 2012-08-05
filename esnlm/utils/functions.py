""" A few useful functions"""

import numpy as np


def softmax(x):
    """Returns an array with each row containing the softmax evaluation of the corresponding row of x.
    
    Parameters
    ----------
    x : an array of size nb_samples*nb_features
    
    Examples
    --------
    >>> import numpy as np
    >>> from esnlm.utils.functions import softmax
    >>> x = np.array([[0.5, 0.5]])
    >>> softmax(x)
    array([[ 0.5,  0.5]])
    """
        
    e = np.exp(x)
    if x.ndim is 1:
        z = np.sum(e, axis=0)
    elif x.ndim is 2:
        z = np.tile(np.expand_dims(np.sum(e, axis=1), axis=1), (1, x.shape[1]))
    
    return e/z

def sparseReservoirMatrix(shape, d):
    """ Returns a sparse reservoir matrix.
    
    Parameters
    ----------
    shape : the shape of the matrix
    d : the density of the matrix
    """
    
    w1 = np.random.rand(shape[0], shape[1])
    w2 = np.array(w1)
    w1[w1<(1-float(d)/2)] = 0.
    w2[w2>float(d)/2] = 0.
    w = w1+w2
    mask = np.array(w)
    mask[mask>0] = 1
    w = mask - 2*w
    w[w>0] = 1
    w[w<0] = -1
    w = 0.97*w/np.amax(np.absolute(np.linalg.eigvals(w)))
    return w