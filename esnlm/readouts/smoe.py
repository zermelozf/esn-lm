""" Supervised Mixture of Experts using scikit.learn's LogisticRegression"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def eucl(n):
    a = int(np.ceil(np.sqrt(n)))
    b = n/a
    c = n % a
    d = [a for k in range(b)]
    if c != 0:
        d.append(c)
    return d
    
def lab(labels, n):
    nlabels = [label/int(np.ceil(np.sqrt(n))) for label in labels]
    mlabels = [label % int(np.ceil(np.sqrt(n))) for label in labels]
    return np.array(nlabels), np.array(mlabels)    
        
class SupervisedMoE:
    """ Implements the supervised mixture of experts model to reduce the training time of the algorithm.
        The trick was introduced by Bengion in Hierarchical Probabilistic Language Models or something. 
        Uses the scikit-learn wrapper of Liblinear's logistic regression.
        TODO: Add references
    """
     
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.experts_dim = eucl(output_dim)
        self.nb_experts = len(self.experts_dim)

        self.gates = LogisticRegression()
        self.experts = []
        for k in range(self.nb_experts):
            self.experts.append(LogisticRegression())
            
    def fit(self, x, labels):
        nlabels, mlabels = lab(labels, self.output_dim)
        print "...... learning gating model",
        self.gates.fit(x, nlabels)
        print "and expert", 
        for i in range(self.nb_experts):
            print i,
            idx = np.nonzero(nlabels==i)[0]
            self.experts[i].fit(x[idx,:], mlabels[idx])
        print "The end."
    
    def py_given_x(self, x):
        c = self.gates.predict_proba(x)
        py = np.zeros((x.shape[0], self.output_dim))
        for i in range(self.nb_experts):
            p = self.experts[i].predict_proba(x)
            pe = np.tile(np.expand_dims(c[:, i], axis=1) , (1, p.shape[1]))
            py[:,sum(self.experts_dim[:i]):sum(self.experts_dim[:i+1])] = pe*p
        return py
    
    def sample_y_given_x(self, x):
        post = self.py_given_x(x)
        y = np.array([np.random.multinomial(1, post[i, :]) for i in range(x.shape[0])])
        return y 
