import unittest

class TestNLP(unittest.TestCase):
        
    def test_word_to_index(self):
        from nlp_utils import word_to_index
        words = ['je', 'suis', 'une', 'mite', '.', 'tu', 'es', 'une', 'mite', '.' ]
        vocabulary = ['je', 'suis', 'une', 'mite', '.' ]
        
        indexes = [word_to_index(word, vocabulary) for word in words]
        assert indexes == [0, 1, 2, 3, 4, 5, 5, 2, 3, 4]
        
        
if __name__=="__main__":
    unittest.main(defaultTest='test_suite')