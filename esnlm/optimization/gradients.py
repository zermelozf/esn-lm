""" Gradient and Hessian for weighted cross-entropy"""

import numpy as np

def gradient(self, x, y, post, w):
    """ Computes the gradient of the cross-entropy.
    
        Parameters
        ----------
        x : input of size nb_samples*nb_features
        y : target of size nb_samples*nb_classes
        post : post
        w : w
    """
        
    bw = np.tile(np.expand_dims(w, axis=1), (1,y.shape[1]))
    grad = np.dot(x.T, bw*(y - post))
    grad = grad.reshape(grad.size)
    return grad
    
def hessian(self, x, y, post, w):
    """ Computes the hessian of the cross-entropy.
    
        Parameters
        ----------
        x : input of size nb_samples*nb_features
        y : target of size nb_samples*nb_classes
        post : post
        w : w
    """
    hessian = np.zeros((x.shape[1]*y.shape[1], x.shape[1]*y.shape[1]))
    p = post
    dim = p.shape[1]
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            wp = np.tile(np.expand_dims(w*x[:,i]*x[:,j], axis=1), (1, dim))*p
            h = np.diag(np.sum(wp, axis=0)) - np.dot(p.T, wp)
            hessian[i*dim:(i+1)*dim,j*dim:(j+1)*dim] = -h
            if j is not i:
                hessian[j*dim:(j+1)*dim,i*dim:(i+1)*dim] = -h
            
    return hessian
