""" Train the supervised mixture of experts model on Elman's grammar. """

import cPickle as pickle
from esnlm.reservoir import sparseReservoirMatrix, buildEsn
from esnlm.features import Features
from esnlm.readouts import SupervisedMoE

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
input_dim = output_dim = len(vocabulary)
features_dim, reservoir_dim = 5, 10

reservoir_matrix = sparseReservoirMatrix((reservoir_dim, reservoir_dim), 0.27)
reservoir = buildEsn(input_dim=features_dim, reservoir_matrix=reservoir_matrix)

features = Features(input_dim, features_dim).learn(utrain, ytrain, reservoir, max_iter=15)

readout = SupervisedMoE(reservoir_dim, output_dim)

print "... building data"
xtrain = reservoir.execute(features[utrain])
xtest  = reservoir.execute(features[utest])

print "... training readout"
readout.fit(xtrain, ytrain)

print "... results"
mpy = readout.py_given_x(xtrain)
perplexity = 1.
for i, p in enumerate(mpy):
    perplexity *= p[ytrain[i]]**(-1./len(ytrain))
print "Training perplexity:", perplexity

mpy = readout.py_given_x(xtest)
perplexity = 1.
for i, p in enumerate(mpy):
    perplexity *= p[ytest[i]]**(-1./len(ytest))
print "Testing perplexity:", perplexity




