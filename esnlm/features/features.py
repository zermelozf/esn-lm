""" Stuff with features """

import numpy as np

class Features:
    """ The class for the pre-recurrent features """
    def __init__(self, nb_features, features_dim):
        self.params = np.random.rand(nb_features, features_dim)
        
    def learn(self, source, target, reservoir, max_iter=15, mode='text'):
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
            readout = np.random.rand(reservoir.output_dim, self.params.shape[0])/reservoir.output_dim
        
            print "... gradient descent:",
            momentum = 0.
            lrdecay = 0.95
            lr = 5.
            
            for t in range(max_iter):
                print t,
                lr *= lrdecay
                for i in range(len(source)):
                    fea = np.expand_dims(self.params[source[i]], axis=0)
                    res = reservoir.execute(fea)
                    out = np.dot(res, readout)
                    delta0 = out - target[i]
                    delta1 = np.dot(delta0, readout.T)*(1 - res**2)
                    gradf = np.dot(delta1, reservoir.w_in)
                    gradr = np.dot(res.T, delta0)
                    
                    self.params[source[i]] += -.005*lr*gradf[:,0]
                    readout += -.001*lr*(gradr + 0.9*momentum)
                    momentum = gradr
        
            print "The end."
            return self.params
    
        if mode == 'sentences':
            """ Learns features reinitializing the reservoir after each sentence """
            from ..reservoir import init_reservoir
            
            features = self.params        
            readout = np.random.rand(reservoir.output_dim, target[0].shape[1])/reservoir.output_dim
            
            print "... gradient descent:",
            momentum = 0.5
            lr_decay = 0.95
            lr = 5.
            for t in range(max_iter):
                initial_state = init_reservoir(reservoir, source, features)
                print t,
                lr *= lr_decay
                for k in range(len(source)):
                    reservoir.states = np.c_[initial_state].T
                    res = reservoir.execute(np.dot(source[k], features))
                    output = np.dot(res, readout)
                    for i in range(output.shape[0]):
                        e = output[i] - target[k][i]
                        p1 = np.dot(e, readout.T)
                        p2 = 1-res[i]**2
                        gradf = np.array([np.dot(p1*p2, reservoir.w_in)])*np.array([source[k][i]]).T
                        features += -.005*lr*gradf
                        gradr = np.array([e])*np.array([res[i]]).T
                        readout += -.001*lr*(gradr + 0.9*momentum)
                        momentum = gradr
            print "The end."
            self.params = features
            return features
        