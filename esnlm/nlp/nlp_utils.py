import numpy as np

def word_to_index(word, vocabulary):
        try:
            return vocabulary.index(word)
        except:
            return len(vocabulary) 
        
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
    
    return np.prod(np.sum(py[:,y]+1e-18, axis=1)**(-1./len(y)))