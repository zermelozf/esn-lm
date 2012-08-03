import unittest
import numpy as np
from ..readouts import LogisticRegression

input_dim, output_dim = 5, 5
x = np.random.rand(50, input_dim-1)
x = np.hstack([x, np.ones((x.shape[0], 1))])
y = LogisticRegression(input_dim, output_dim).sample_y_given_x(x)

class TestLogisticRegression(unittest.TestCase):
        
    def testSampling(self):
        assert (np.sum(y, axis=1) == 1.).all()
        
    def testFitting(self):
        model = LogisticRegression(input_dim, output_dim)
        ll_before = model.log_likelihood(x, y)
        model.fit(x, y, method='Newton-Raphson', max_iter=20)
        ll_after = model.log_likelihood(x, y)
        
        assert ll_after > ll_before
        
        
        
if __name__=="__main__":
    unittest.main(defaultTest='test_suite')