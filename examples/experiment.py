""" Compare the different models """

import os
from datetime import datetime
import numpy as np
import cPickle as pickle
import Oger as og
from esnlm.features import Features
from esnlm.readouts import LinearRegression, MixtureOfExperts, SupervisedMoE
#from esnlm.readouts import LogisticRegression
from sklearn.linear_model import LogisticRegression
from esnlm.utils import perplexity
from time import time

def run_experiment(data, options):
    reservoir = og.nodes.ReservoirNode( input_dim       =   data['udim'],
                                        output_dim      =   options['reservoir_dim'],
                                        spectral_radius =   options['spectral_radius'])
    
    if options['with_features'] == True:
        fnode = Features(input_dim, options['features_dim'])
        features = fnode.learn(utrain, ytrain, reservoir, max_iter=options['features_max_iter'])
    else:
        features = np.eye(data['udim'])
    
    readouts = [LinearRegression(options['reservoir_dim'], data['ydim']),
                LogisticRegression(),
                #LogisticRegression(options['reservoir_dim'], data['ydim']),
                SupervisedMoE(options['reservoir_dim'], data['ydim']),
                MixtureOfExperts(input_dim=options['reservoir_dim'], nb_experts=options['nb_experts'], output_dim=data['ydim'])
                ]
    data['xtrain'] = reservoir.execute(features[utrain])
    data['xtest']  = reservoir.execute(features[utest])
    
    perp = []
    t = []
    for readout in readouts:
        print "... learning", readout
        tstart = time()
        try:
            readout.fit(data['xtrain'], data['ytrain'], method='CG', max_iter=options['NR_max_iter'])
        except:
            readout.fit(data['xtrain'], data['ytrain'])
        try:
            perp.append(perplexity(readout.py_given_x(data['xtest']), data['ytest']))
        except:
            perp.append(perplexity(readout.predict_proba(data['xtest']), data['ytest']))
        t.append(-tstart + time())
        
    path = '../results/'+str(datetime.now())+'/'
    save_experiment(options, data, features, reservoir, readouts, perp, t, path)

def save_experiment(options, data, features, reservoir, readouts, perp, t, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'results', 'w') as f:
        pickle.dump({'options':options,
                     'data':data, 
                     'features':features, 
                     'reservoir':reservoir, 
                     'readouts':readouts, 
                     'perplexity':perp}, f, protocol=0)
    with open(path+'description.txt', 'w') as f:
        import csv
        w = csv.writer(f)
        for key, val in options.items():
            w.writerow([key, val])
    
    with open(path+'results.txt', 'w') as f:
        for i, readout in enumerate(readouts):
            f.write(str(readout) + '\t' + str(perp[i]) + '\t' + str(t[i]) + '\n')

if __name__ == "__main__":
    
#    options = {}
#    options['train_dataset']    = './../datasets/t5_train'
#    options['test_dataset']     = './../datasets/t5_test'
#    options['with_features']    = False, True
#    options['features_dim']     = 2, 3, 5, 10
#    options['reservoir_dim']    = 5, 10, 15, 25, 50, 100
#    options['spectral_radius']  = 0.97
#    options['nb_experts']       = 2, 3, 5
#    options['features_max_iter'] = 10
#    options['NR_max_iter']      = 15
    
    options = {}
    options['train_dataset']    = './../datasets/t5_train'
    options['test_dataset']     = './../datasets/t5_test'
    options['with_features']    = False
    options['features_dim']     = 2
    options['reservoir_dim']    = 5
    options['spectral_radius']  = 0.97
    options['nb_experts']       = 2
    options['features_max_iter'] = 10
    options['NR_max_iter']      = 25
    
    print "... loading text"
    with open(options['train_dataset']) as f:
        text_train = pickle.load(f)   
    with open(options['test_dataset']) as f:
        text_test = pickle.load(f)  
    vocabulary = list(set(text_train))
    
    print "... building data"
    utrain = [vocabulary.index(w) for w in text_train[:-1]]
    ytrain = [vocabulary.index(w) for w in text_train[1:]]
    utest = [vocabulary.index(w) for w in text_test[:-1]]
    ytest = [vocabulary.index(w) for w in text_test[1:]]
    input_dim = output_dim = len(vocabulary)
    
    data = {}
    data['udim'] = input_dim
    data['ydim'] = output_dim
    data['utrain'] = utrain
    data['ytrain'] = ytrain
    data['utest'] = utest
    data['ytest'] = ytest

    print "... running experiment"
    run_experiment(data, options)
    
    