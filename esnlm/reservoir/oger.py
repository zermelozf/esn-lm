""" Reservoir stuff using Oger """

import numpy as np
import Oger as og

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
    w = 0.97*w/og.utils.get_spectral_radius(w)
    return w

def init_reservoir(reservoir, source, features=None):
    """ Initialize the reservoir to a state after convergence """
    if features == None:
        features = np.eye(source[0].shape[1])
        
    for i in range(20):
        reservoir.execute(np.dot(source[i], features))

    return reservoir.states[-2]

def build_esn(input_dim, reservoir_matrix):
    """ Build an ESN with fixed parameters using Oger. """
    reservoir = og.nodes.ReservoirNode(input_dim=input_dim,
                                       output_dim=reservoir_matrix.shape[0],
                                       w=reservoir_matrix,
                                       input_scaling=1.,
                                       reset_states=False,
                                       spectral_radius=0.97)
    return reservoir

def esn_data(source, target, reservoir, init_state, features=None, mode='text'):
    """ Builds a dataset of reservoir activations """
    
    if features == None:
        features = np.eye(source[0].shape[1])

    initial_state = init_state
    
    x = []
    if mode == 'sentences':
        for i in range(len(source)):
            reservoir.states = np.c_[initial_state].T
            x.append(reservoir.execute(np.dot(source[i], features)))
        x = np.vstack(x)
    
    if mode == 'text':
        x = reservoir.execute(features[source])
    
    return x

def esn_data2(u, y, reservoir, x0, features=None, mode='text'):
    """ Builds a dataset of reservoir activations """
    
    if features == None:
        features = np.eye(u[0].shape[1])
    
    x = []
    if mode == 'sentences':
        for i in range(len(u)):
            reservoir.states = np.c_[x0].T
            x.append(reservoir.execute(features[u[i]]))
        x = np.vstack(x)
    
    if mode == 'text':
        x = reservoir.execute(features[u])
    
    return x


