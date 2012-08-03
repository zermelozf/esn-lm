=========
Reference
=========

This is the class and function reference of esn-lm.

.. contents:: List of modules
   :local:


:mod:`esnlm.utils`: Utilities
=============================

.. automodule:: esnlm.utils
	:members: softmax
	
:mod:`esnlm.reservoir`: Reservoir
=============================

.. automodule:: esnlm.reservoir
	:members: sparseReservoirMatrix, init_reservoir, build_esn
	
:mod:`esnlm.nlp`: Text processing
=================================

.. automodule:: esnlm.nlp
	:members: setVocLut, similarity, perplexity, sentences_ngram, word_distrib, to_num, load_sentences, load_train_test


:mod:`esnlm.readouts`: Readouts
===============================

.. automodule:: esnlm.readouts
	:members: LogisticRegression, MixtureOfExperts, SupervisedMoE
	
:mod:`esnlm.features`: Features
===============================

.. automodule:: esnlm.features
	:members: Features
	
:mod:`esnlm.optimization`: Optimization
=======================================

.. automodule:: esnlm.optimization
	:members: gradient, hessian, newton_raphson, expectation_maximization, expectation_maximization2
	
