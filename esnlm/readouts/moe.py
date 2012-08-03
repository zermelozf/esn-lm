import numpy as np
from logistic import LogisticRegression
from ..optimization import expectation_maximization

#### CLASS DEFINITION
class MixtureOfExperts:
    """ The Mixture of Experts model"""
    
    def __init__(self, input_dim, nb_experts, output_dim):
        self.nb_experts = nb_experts
        self.output_dim = output_dim
        self.gates = LogisticRegression(input_dim, nb_experts)
        self.experts = [LogisticRegression(input_dim, output_dim) for k in range(nb_experts)]
    
    def pz_given_x(self, x):
        return self.gates.py_given_x(x)
        
    def py_given_x(self, x):
        pz = self.gates.py_given_x(x)
        py = np.zeros((x.shape[0], self.output_dim))
        for z in range(self.nb_experts):
            pzb = np.tile(np.expand_dims(pz[:, z], axis=1), (1, self.output_dim))
            py += pzb*self.experts[z].py_given_x(x)
        return py
    
    def py_given_xz(self, x, z):
        return self.experts[z].py_given_x(x)
    
    def lik_y_for_every_z(self, x, y):
        py = np.zeros((x.shape[0], self.nb_experts))
        for z in range(self.nb_experts):
            py[:, z] = np.sum(y*self.py_given_xz(x, z), axis=1)
        return py
    
    def pz_given_xy(self, x, y):
        pz_given_x = self.pz_given_x(x)
        lik_y_forallz = self.lik_y_for_every_z(x, y)
        pz_given_xy = lik_y_forallz*pz_given_x
        renorm = np.tile(np.expand_dims(np.sum(pz_given_xy, axis=1), axis=1), (1, pz_given_xy.shape[1]))
        pz_given_xy = pz_given_xy/renorm
        return pz_given_xy
        
    
    def sample_y_given_x(self,x):
        py = self.py_given_x(x)
        y = np.array([np.random.multinomial(1,py[i,:]) for i in range(x.shape[0])])
        return y
    
    def log_likelihood(self, x, y):
        lik_y = self.lik_y_for_every_z(x, y)
        pz_given_x = self.pz_given_x(x)
        return np.sum(np.log(np.sum(pz_given_x*lik_y, 1)))
    
    def fit(self, x, y, method='CG', max_iter=100):
        """ The model is trained using Generalized Expectation-Maximization.
            In the Maximization step the Conjugate-Gradient algorithm provided by scipy.optimize is used
            by default.
        """
        
        ll, Q1, Q2 = expectation_maximization(self, x, y, max_iter=max_iter)
        return ll, Q1, Q2
        
        
        