import nltk
from esnlm.readouts import SupervisedMoE
from esnlm.reservoir import sparseReservoirMatrix, build_esn
from esnlm.features import Features

emma = nltk.corpus.gutenberg.raw('austen-emma.txt')[:50000]

### Analyse symbol frequency
fd = nltk.FreqDist()
for s in emma:
    fd.inc(s)

### Map words to labels
vocabulary = list(set(emma))
quant = [fd.keys().index(s) for s in vocabulary]
lut = set(zip(quant, vocabulary))
lut = sorted(lut, key=lambda el: el[0])

label, vocabulary = zip(*lut)

### Relabel 
labels = [label[vocabulary.index(word)] for word in emma]
labels = [min(l, 48) for l in labels]

print "... building dataset"
### Training set
vocabulary_size = 49 

u1, y1 = labels[:20000], labels[:20000]
u2, y2 = labels[20000:40001], labels[20000:40001]

input_dim, features_dim, reservoir_dim, output_dim = vocabulary_size, 2, 100, vocabulary_size

### Reservoir
reservoir_matrix = sparseReservoirMatrix((reservoir_dim,reservoir_dim), 0.27)
reservoir = build_esn(features_dim, reservoir_matrix)

### Features
features = Features(input_dim, features_dim).learn(u1, y1, reservoir, max_iter=0)

### Readout
m = SupervisedMoE(input_dim, output_dim)

### Data
x1 = reservoir.execute(features[u1])
x2 = reservoir.execute(features[u2])

print ".. training"
m.fit(x1, y1)

print "... results"
mpy = m.py_given_x(x1)

perplexity = 1
for i, p in enumerate(mpy):
    perplexity *= p[y1[i]]**(-1./len(y1))
print "Perplexity:", perplexity

mpy = m.py_given_x(x2)
perplexity = 1
for i, p in enumerate(mpy):
    perplexity *= p[y2[i]]**(-1./len(y1))
print "Perplexity:", perplexity
mpy = None

import numpy as np
y = m.sample_y_given_x(x2)
lb = np.nonzero(y==1.)[1]

print lb

syms = [vocabulary[label.index(l)] for l in lb]
print ''.join(syms)






