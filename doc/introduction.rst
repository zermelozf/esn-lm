Introduction
============

Machine translation, speech recognition, orthographic error correction,
input systems for Japanese or Chinese... These are only a few examples
among the many systems that rely on probabilistic language modeling.
With the explosion of linguistic data available on the Internet or
that can be obtained from huge corpora such as the British National
Corpus, the study of the statistical properties of the language has
become much easier. 

Simple models such as the :math:`n`-gram model can be trained efficiently
using smoothing techniques \cite{Chen:1996:ESS:981863.981904} and
allow to quickly building an accurate language model that can then
be used as a sub-part in a more complex systems. However, these simple
language models are criticized for they only capture superficial linguistic
structures and are unable to take into account long term dependencies
that occur in natural languages. Moreover, :math:`n`-gram models suffer
from the curse of dimensionality: for a vocabulary containing :math:`|V|`
words, there are :math:`|V|^{n}` possible different sequences of words
and thus :math:`n`-grams require :math:`|V|^{n}` parameters. Since most of the
possible combinations of words are never seen during the training
phase, the parameters cannot be estimated properly using maximum likelihood.
Although different smoothing techniques allow facing this data sparsity,
they usually rely on simple notions of similarity to generalize. For
example back-off :math:`n`-gram models rely on the frequency of shorter
sequences.

Recently, several neural language models have been proposed \cite{Mnih2007,Bengio2009,collobert:2008}.
They address the representational power issue of :math:`n`-gram models
and capture syntactic and semantic similarity between words by introducing
distributed representation both for a word and its context. Using
neural networks for language modeling is not a recent idea but until
a few years ago, statistical language models were predominantly variations
of :math:`n`-grams models. The main drawback of neural language models
is their training time. However, since the fundamental operation in
a neural network is matrix multiplication, several techniques (such
as GPU computation) can be used to greatly reduce the time needed
by the algorithm.

Three types of networks have been applied to natural language processing
tasks:


* 	Feed-forward networks \cite{Bengio:2003:NPL:944919.944966,collobert:2008}
	are fed with a finite number :math:`k` of words at one time and predict
	the next word according to that finite context. Usually, those models
	are time-delay neural networks fed with a big vector of :math:`k` concatenated
	word vectors. These models are close to :math:`n`-grams since they take
	into account only a finite context but can develop very complex distributed
	representations and automatically perform smoothing.

* 	Recurrent neural networks \cite{Elman:1991:DRS:125342.125347,Mikolov2011a,Mikolov2011}
	can take an arbitrarily long context into account when predicting
	the next word. They are fed with one word at a time and develop a
	representation of the past input sequence in their recurrent layer.
	They are usually trained with back-propagation through time which
	makes them slower than feed-forward networks to train. Recurrent neural
	networks perform at the state of the art level in word recognition
	in speech processing \cite{Mikolov2011a}.

* 	Recursive neural networks learn compositional representation of words
	and phrases by applying recursively the same neural network. Like
	recurrent neural networks, recursive neural networks share their parameters.
	However they allow more flexibility by having a tree like hierarchical
	structure when recurrent neural networks only merge the representation
	of words and context as time passes. They obtain excellent results
	on several natural language processing tasks despite having been introduced
	very recently \cite{Socher2011,Socher2010}.

These neural network language models not only outperform :math:`n`-gram
at the task of predicting the next word in a sentence but are also
able to provide representation of words, sentences and text that capture
high level features that sometimes have an intuitive explanation.

Although each neural model has its own strength and weakness, recurrent
neural network are of particular interest to us. They allow to elegantly
taking into account arbitrarily long dependency between words and
can represent very compactly the context of a sentence or a document
in their recurrent layer. More specifically, in this thesis we investigate
echo state networks: a recent development in the field of recurrent
neural networks belonging to a collection of techniques called Reservoir
Computing \cite{Reservoir_computing}. The philosophy adopted in Reservoir
Computing is to consider the recurrent layer as a large reservoir
of nonlinear transformations of the input data and decouple the learning
of parameters inside and outside the reservoir. Echo state networks
have already shown promising performance on time series prediction
tasks but have seldom been used in more abstract settings such as
natural language modeling. Starting from the results obtained in \cite{Tong:2007:SIL:1265618.1265889},
we investigate the properties of echo state networks when applied
to language modeling tasks. 

In this thesis, our goal is not only to improve the accuracy of the
language model but also to extract features and representation that
are useful and allow understanding better the underlying mechanisms
of the networks. Moreover, we try to keep the complexity of the learning
algorithms used to obtain representations and train the different
components of the model at a minimum so that it may be possible to
scale the result obtained to larger datasets. The main contribution
of this document is to explicitly describe the kind of representations
developed by echo state networks for natural language modeling and
propose modifications of the conventional architecture to leverage
this representation.