""" Stuff with features """

import numpy as np

class Features:
    """ The class for the pre-recurrent features """
    def __init__(self, nb_features, features_dim):
        self.params = np.random.rand(nb_features, features_dim)
        
    def learn(self, u, y, reservoir, max_iter=15, mode='text', verbose=False):
        """ Learn the features for a simple linear readout function 
            
            Parameters
            ----------
            source : the input to the reservoir, i.e one-hot word representations
            target : the target output, i.e. the next word to predict
            reservoir : the reservoir the add pre-recurrent features to
            
            Returns
            -------
            The features that are learned.
        """
        
        if mode == 'text':
            features = self.params
            readout = np.random.rand(reservoir.output_dim, self.params.shape[0])/reservoir.output_dim
        
            print "... gradient descent on features:",
            momentum = 0.
            lrdecay = 0.95
            lr = 5.
            
            for t in range(max_iter):
                print t,
                lr *= lrdecay
                for i in range(len(u)):
                    fea = np.expand_dims(features[u[i]], axis=0)
                    tar = np.zeros((1, features.shape[0]))
                    tar[:, y[i]] = 1.
                    res = reservoir.execute(fea)
                    out = np.dot(res, readout)
                    delta0 = out - tar
                    delta1 = np.dot(delta0, readout.T)*(1 - res**2)
                    gradf = np.dot(delta1, reservoir.w_in)
                    gradr = np.dot(res.T, delta0)
                    
                    features[u[i]] += -.005*lr*gradf[0]
                    readout += -.001*lr*(gradr + 0.9*momentum)
                    momentum = gradr
        
            print "The end."
            return self.params
    
        