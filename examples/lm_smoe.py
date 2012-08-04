import numpy as np
from esnlm.readouts import SupervisedMoE
from esnlm.nlp import to_num, load_train_test, word_distrib, similarity, perplexity
from esnlm.reservoir import sparseReservoirMatrix, build_esn, init_reservoir, esn_data
from esnlm.features import Features

    
mode = 'text'
dim = 100
sentences1, sentences2 = load_train_test(filename='../datasets/t5')

distrib1 = word_distrib(sentences1)
distrib2 = word_distrib(sentences2)

#Share the reservoir matrix
reservoir_matrix = sparseReservoirMatrix((dim,dim), 0.27)

#Build train and test data without features
#    sreservoir = build_esn(24, reservoir_matrix)
#    initial_state = init_reservoir(sreservoir, sentences1)
#    u1, x1, y1 = esn_data(sentences1, sreservoir, initial_state, mode=mode)
#    u2, x2, y2 = esn_data(sentences2, sreservoir, initial_state, mode=mode)

#    Build train and test data with features
print "... building dataset"
u1, y1 = to_num(sentences1)
u2, y2 = to_num(sentences2)
nb_features, features_dim = u1[0].shape[1], 2

if mode == 'text':
    u1 = np.nonzero(np.vstack(u1)==1.)[1]
    y1 = np.vstack(y1)
    u2 = np.nonzero(np.vstack(u2)==1.)[1]
    y2 = np.vstack(y2)

freservoir = build_esn(features_dim, reservoir_matrix)
features = Features(nb_features, features_dim).learn(u1, y1, freservoir, max_iter=15, mode=mode, verbose=True)

initial_state = init_reservoir(u1, freservoir, features)
x1 = esn_data(u1, y2, freservoir, initial_state, features, mode=mode)
x2 = esn_data(u2, y2, freservoir, initial_state, features, mode=mode)

input_dim, output_dim = x1.shape[1], y1.shape[1]

m = SupervisedMoE(input_dim, output_dim)

ylabels = np.nonzero(y1 != 0)[1]

print ".. training"
m.fit(x1, ylabels)

K_soft_distrib1 = m.py_given_x(x1)
sim1 = np.mean(similarity(distrib1, K_soft_distrib1))
K_soft_distrib2 = m.py_given_x(x2)
sim2 = np.mean(similarity(distrib2, K_soft_distrib2))

print "Similarity:", sim1, sim2
print "Perplexity:", perplexity(K_soft_distrib1, y1), perplexity(K_soft_distrib2, y2)