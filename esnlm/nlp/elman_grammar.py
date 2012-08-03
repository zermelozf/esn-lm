import numpy as np
import nltk

### NLP UTILS
vocabulary = ['John','Mary','boy','girl','cat','dog','boys','girls','cats','dogs',
              'feeds','walks','lives','hits','sees','hears',
              'feed','walk','live','hit','see','hear',
              'who','.']

lut = dict([(vocabulary[i],  1.*np.eye(len(vocabulary))[i]) for i in range(len(vocabulary))])

def setVocLut():
    return vocabulary, lut

def similarity(d1, d2):
    """ Returns the similarity between two arrays.
    
    Parameters
    ----------
    d1 : array of size m*n
    d2 : array of size m*n
    
    Returns
    -------
    cos : array of size m*n with each row corresponding to the cosine product of the corresponding
    rows of d1 and d2
    
    Notes
    -----
    TODO: modify so that big arrays can be compared
    
    """
        
    normx = np.sqrt(np.sum(np.array(d1)**2, axis=1))
    normy = np.sqrt(np.sum(np.array(d2)**2, axis=1))
    prod = np.sum(np.array(d1)*np.array(d2),axis=1)
    return prod/(normx*normy)

def perplexity(py, y):
    """ Returns the perplexity of py on y 
        
        Parameters
        ----------
        py : array of size m*n with py[i,j] containing the probability of y[i,j]
        y : array of size m*n with one-hot rows.
        
        Returns
        -------
        perplexity : TODO math_formula
        
        Notes
        -----
        TODO: modify for big arrays.
    """
    return np.prod(np.sum(py*y+1e-18, axis=1)**(-1./y.shape[0]))


### Text as sentences collection
def sentences_ngrams(sentences):
    """ Returns the n-grams of a collection of sentences """
    ingrams = []
    for s in sentences:
        for l in range(len(s)+1):
            ingrams.extend(list(nltk.util.ingrams(s,l)))
        
    ngrams = set(ingrams)
    freq = nltk.FreqDist(ingrams)
    return ngrams, freq

def word_distrib(sentences):
    """ Returns the probability distribution of words in context for a collection of sentences """
    ngrams, freq = sentences_ngrams(sentences)

    pos = []
    distrib = []
    for s in sentences:
        for i in range(len(s)-1):
            t1 = tuple(s[0:i+1])
            z = freq.freq(t1)
            if i == 0:
                z = z/4
            x = []
            for v in vocabulary:
                t2 = list(t1)
                t2.append(v)
                x.append(freq.freq(tuple(t2))/z)
            pos.append(t1)
            distrib.append(x)
    
    assert len(pos) == len(distrib)
    return np.r_[distrib]

def to_num(sentences):
    """ Convert sentences to a numerical array """
    vocabulary, lut = setVocLut()
    
    source = []
    target = []
    for s in sentences:
        words = []
        nwords = []
        for k in range(len(s)-1):
            words.append(lut[s[k]])
            nwords.append(lut[s[k+1]])
        source.append(np.r_[words])
        target.append(np.r_[nwords])
    assert len(source) == len(sentences)
    return source, target


### Load Elman grammar's datasets
import cPickle as pickle

def load_sentences(filename='../../datasets/tong6words.enum'):
    """ Load the sentences contained in an enumeration file. Adds a stop mark at the begining of each sentence 
    
    Examples
    --------
    Sequence::
        ['Mary', 'sees', 'cats', '.'] -> ['.', 'Mary', 'sees', 'cats', '.']
    """
    print "... loading", 
    f = open(filename)
    sentences = pickle.load(f)
    for k in range(len(sentences)):
        sentences[k].insert(0,'.')
    print len(sentences), "sentences",
    print "(", sum([len(s) for s in sentences]), "words )"
    
    return sentences

def load_train_test(filename='../../datasets/t5'):
    """ Load the sentences contained in an enumeration file and seperated into a training
    and a testing set. Adds a stop mark at the begining of each sentence 
    
    Examples
    --------
    Sequence::
        ['Mary', 'sees', 'cats', '.'] -> ['.', 'Mary', 'sees', 'cats', '.']
    """
    print "... loading", 
    f = open(filename)
    [sentences1, sentences2] = pickle.load(f)
    
    s1 = []
    for k in range(len(sentences1)):
        s = list(sentences1[k])
        s.insert(0,'.')
        s1.append(s)
    
    s2 = []
    for k in range(len(sentences2)):
        s = list(sentences2[k])
        s.insert(0,'.')
        s2.append(s)
    
    print np.sum([len(s) for s in s1]), np.sum([len(s) for s in s2]), "words"
    
    return s1, s2

