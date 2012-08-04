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
    w = 0.97*w/np.amax(np.absolute(np.linalg.eigvals(w)))
    return w

def init_reservoir(u, reservoir, features=None):
    """ Initialize the reservoir to a state after convergence """
    if features == None:
        features = np.eye(u[0].shape[1])
    
    for s in u[:min(10,len(u))]:
        reservoir.execute(features[s[:min(s.size, 100)]])
        if s.size > 100:
            break

    return reservoir.states[-2]

def buildEsn(input_dim, reservoir_matrix):
    """ Build an ESN with fixed parameters using Oger. """
    reservoir = og.nodes.ReservoirNode(input_dim=input_dim,
                                       output_dim=reservoir_matrix.shape[0],
                                       w=reservoir_matrix,
                                       input_scaling=1.,
                                       reset_states=False,
                                       spectral_radius=0.97)
    return reservoir
