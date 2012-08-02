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