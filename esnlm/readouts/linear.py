""" The conventional linear readout """

import numpy as np

class LinearRegression:
    def __init__(self, input_dim, output_dim):
        self.params = np.random.rand(input_dim, output_dim)
    
    def py_given_x(self, x):
        py = np.dot(x, self.params)
        py[np.nonzero(py < 0)] = 1e-3
        py = py/np.tile(np.expand_dims(np.sum(py, axis=1), axis=1), (1, py.shape[1]))
        return py
    
    def sample_y_given_x(self, x):
        py = self.py_given_x(x)
        y = np.array([np.random.multinomial(1,py[i,:]) for i in range(x.shape[0])])
        return y
    
    def fit(self, x, y):
        y = np.eye(self.params.shape[1])[y]
        self.params = np.linalg.lstsq(x, y)[0]
        return self.params