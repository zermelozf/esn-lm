""" Compare the different models """

import cPickle as pickle
import Oger as og
from esnlm.features import Features
from esnlm.utils import compare
from esnlm.readouts import *

print "... loading text"
with open('./../datasets/t5_train') as f:
    text_train = pickle.load(f)
    
with open('./../datasets/t5_test') as f:
    text_test = pickle.load(f)
    
vocabulary = list(set(text_train))

### Transform text into labels
utrain = [vocabulary.index(w) for w in text_train[:-1]]
ytrain = [vocabulary.index(w) for w in text_train[1:]]

utest = [vocabulary.index(w) for w in text_test[:-1]]
ytest = [vocabulary.index(w) for w in text_test[1:]]

print "... building model"
### Hyperparameters
input_dim = output_dim = len(vocabulary)
reservoir_dim = 25
spectral_radius = 0.97

print "WITHOUT FEATURES:"
import numpy as np
features = np.eye(input_dim)
reservoir = og.nodes.ReservoirNode( input_dim       =   input_dim,
                                    output_dim      =   reservoir_dim,
                                    spectral_radius =   spectral_radius)
readouts = [LinearRegression(reservoir_dim, output_dim),
            LogisticRegression(reservoir_dim, output_dim),
            SupervisedMoE(reservoir_dim, output_dim),
            MixtureOfExperts(input_dim=reservoir_dim, nb_experts=5, output_dim=output_dim)
            ]
perplexity = compare(features, reservoir, readouts, utrain, ytrain, utest, ytest)

print "WITH FEATURES:"
features_dim = 2
reservoir = og.nodes.ReservoirNode( input_dim       =   features_dim,
                                    output_dim      =   reservoir_dim,
                                    spectral_radius =   spectral_radius)
features = Features(input_dim, features_dim).learn(utrain, ytrain, reservoir, max_iter=10)
readouts = [LinearRegression(reservoir_dim, output_dim),
            LogisticRegression(reservoir_dim, output_dim),
            SupervisedMoE(reservoir_dim, output_dim),
            MixtureOfExperts(input_dim=reservoir_dim, nb_experts=5, output_dim=output_dim)
            ]
fperplexity = compare(features, reservoir, readouts, utrain, ytrain, utest, ytest)

################
### PLOTTING ###
################
print perplexity
print fperplexity

from matplotlib.pyplot import plot, show, legend
p1 = plot(perplexity, color='blue')
p2 = plot(fperplexity, color='red')
legend([p1[0], p2[0]], ['one-hot', 'features'])

show()


