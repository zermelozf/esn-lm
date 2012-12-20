import cPickle as pickle
import random
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist
import nltk

print "... loading text"
text_train = list(nltk.corpus.gutenberg.words('austen-emma.txt'))
print len(set(text_train))
text_test = list(nltk.corpus.gutenberg.words('austen-sense.txt'))

#with open('./../datasets/t5_train') as f:
#    text_train =(' '.join(pickle.load(f))).split(' . ')
#    random.shuffle(text_train)
#    text_train = (' . '.join(text_train)).split(' ')
#    
#with open('./../datasets/t5_test') as f:
#    text_test =(' '.join(pickle.load(f))).split(' . ')
#    random.shuffle(text_test)
#    text_test = (' . '.join(text_test)).split(' ')

print "... training model"
estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2) 
lm = NgramModel(3, text_train, estimator=estimator)

print "... results"
print lm.generate(50, ['dog'])
print lm.perplexity(text_test)
print lm.entropy(text_test)
