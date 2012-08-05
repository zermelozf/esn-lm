import unittest
import numpy as np
from functions import softmax

class Test(unittest.TestCase):

    def test_softmax(self):
        assert (softmax(np.ones(10)) == np.ones(10)/10).all()
        assert (softmax(np.ones((1, 10))) == np.ones((1, 10))/10).all()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()