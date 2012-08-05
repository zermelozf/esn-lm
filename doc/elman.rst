Training on Elman's grammar
===========================

This document describes how to train the esn language model on Elman's grammar.

First import necessary things::

	""" Train the supervised mixture of experts model on Elman's grammar. """
	
	import cPickle as pickle
	import Oger as og
	from esnlm.features import Features
	from esnlm.readouts import *


Then load the training and testing datasets. These files contain sentences generated from Elman's grammar.
The sentences are concateneted and word-tokenized in a long list of words::

	with open('./../datasets/t5_train') as f:
	    text_train = pickle.load(f)
	    
	with open('./../datasets/t5_test') as f:
	    text_test = pickle.load(f)
	    
	
The training and testing text are then transformed into a list of integers
corresponding to the indexes of the words::
 
 	vocabulary = list(set(text_train))
 	
	utrain = [vocabulary.index(w) for w in text_train[:-1]]
	ytrain = [vocabulary.index(w) for w in text_train[1:]]
	
	utest = [vocabulary.index(w) for w in text_test[:-1]]
	ytest = [vocabulary.index(w) for w in text_test[1:]]

Then the model is constructed. It is composed of a feature layer, a reservoir layer, and a readout layer. 
The feature	may or may not be trained and the readout can be chosen among linear, logistic, 
mixture of experts or supervised mixture of experts::

	print "... building model"
	### Hyperparameters
	input_dim = output_dim = len(vocabulary)
	features_dim, reservoir_dim = 2, 25
	spectral_radius = 0.97
	
	### Modules
	reservoir = og.nodes.ReservoirNode( input_dim       =   features_dim,
	                                    output_dim      =   reservoir_dim,
	                                    spectral_radius =   spectral_radius)
	
	features = Features(input_dim, features_dim).learn(utrain, ytrain, reservoir, max_iter=10)
	
	
	#readout = LinearRegression(reservoir_dim, output_dim)
	#readout = LogisticRegression(reservoir_dim, output_dim)
	#readout = MixtureOfExperts(input_dim=reservoir_dim, nb_experts=3, output_dim=output_dim)
	readout = SupervisedMoE(reservoir_dim, output_dim)

In order to train the readout, a dataset of reservoir activation is collected::
	
	xtrain = reservoir.execute(features[utrain])
	xtest  = reservoir.execute(features[utest])
	
	print "... training readout"
	readout.fit(xtrain, ytrain)

Finally the performace of the model is asserted using its perplexity on test data::
	
	def perplexity(py, y):
	    perplexity = 1.
	    for i, p in enumerate(py):
	        perplexity *= p[y[i]]**(-1./len(y))
	    return perplexity 
	
	print "Training perplexity:", perplexity(readout.py_given_x(xtrain), ytrain)
	print "Testing perplexity:",  perplexity(readout.py_given_x(xtest), ytest)

	
The entire code can be found in esn-lm/examples.