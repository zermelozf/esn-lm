Probabilistic Language Models
=============================


Definition
----------

Language models are useful for many natural language applications
such as speech recognition, document classification, translation,
compression, etc. A probabilistic language model assumes natural language
is a stochastic process. It is further assumed that this process is
stationary, meaning that the productions of a language have an invariant
probability distribution as time passes. While this assumption is
clearly incorrect since languages are know to evolve according to
trends in a society, it is thought to be a good approximation of the
reality for a short period of time. These assumptions allow describing
the process of learning a language as equivalent to learning the underlying
symbolic dynamical system that produces speech, text, etc [AndrewsThesis]_.
This statement argues in favor of the use of model such as echo state
network that have enough computational power to learn a large class
of dynamical systems [Lukosevicius2007]_.

The goal of a probabilistic language model is to assign a probability
to every possible sequence of word. This task cannot be realized in
practice since the number of sequences of word that can be spoken
or written is extremely vast. A convenient approach is to factor the
probability of a sequence using the Bayesian chain rule:

.. math::
	\begin{equation}
	P(w_{1}w_{2}w_{3}\ldots w_{K})=\prod_{k=1}^{K}P(w_{n}|w_{1}w_{2}w_{3}\ldots w_{k-1}).
	\end{equation}


However this does not solve the problem in any way since the context
of a word can be arbitrarily long. To address this problem, a very
common simplification is to model the probability of a word taking
into account only a given finite context. This family of language
models is called the :math:`n`-grams family. 


Examples
--------

As discussed previously, the most famous probabilistic model for natural
languages is the :math:`n`-gram model. :math:`n`-gram models try to model the
process underlying language production by an :math:`(n-1)^{th}` order Markov
model. The probability of a word sequence is simplified as follows:

.. math::
	\begin{equation}
	P(w_{1}w_{2}w_{3}\ldots w_{K})=\prod_{k=1}^{K}P(w_{k}|w_{k-(n-1)}w_{k-(n-2)}\ldots w_{k-1})
	\end{equation}

for a :math:`n`-gram model. Estimating the parameters of this model can
be done simply by computing the frequency of every word in a fixed
vocabulary after each context of length up to :math:`n-1`, which is equivalent
to maximizing the likelihood of the corpus it is trained on.

.. math::
	\begin{equation}
	P(w_{i}|w_{i-(n-1)}\ldots w_{i-1})=\frac{P(w_{i}|w_{i-(n-1)}\ldots w_{i-1})}{P(w_{i-(n-1)}\ldots w_{i-1})}.
	\end{equation}


However this maximum likelihood estimate can rarely be used on test
data since some rare context that have never been seen in the training
set are automatically assigned a zero probability. To avoid a division
by zero, several smoothing techniques such as Good-Turing or Kneyser-Ney
[Chen1996]_ have been devised that redistribute
the probability mass in a clever way and allow to assign a probability
to sequences that do not appear in the training set.

Recently a few neural language models [Bengio2003]_, [Mnih08ascalable]_, [Mnih2007]_
have been proposed which perform as well or even better than :math:`n`-gram
models. They address both the problems of the representational power
of Markov models and data sparsity by introducing a distributed representation
for words that capture the similarities between them [Turian2010]_.
In chapter \ref{chap:Adding-a-pre-recurrent} we present an echo state
network augmented with a pre-recurrent feature layer and the algorithm
used to train it.

Another type of language model that has received a lot of attention
recently is the Latent Dirichlet Allocation (LDA) model [Blei2003]_.
This model can be seen as an extension of the probabilistic latent
semantic indexing model (figure \ref{fig:Hierarchical-probabilistic-langu}).
It has been followed by a number of improvements that have been widely
applied to modeling the semantic content of documents. The idea of
these models us to use hierarchical probabilistic graphical models
to capture information about the topic of different documents. While
the original LDA model uses a unigram model as the core language model,
more recent implementation have tried to make use of more complex
models such as hidden Markov models [Andrews2009]_ or even syntactic
parsing trees [Boyd-graber2009]_. 

In chapter \ref{chap:Multiple-Readouts} we present a hierarchical
model for selecting the readout that also tries to capture high level
features. Actually, the latent variable in the mixture of experts
model could be chosen to reflect the topic of the document (see \ref{sec:Supervised-Mixture-of}).

FIGURE

Measure of performance
----------------------

A probabilistic language model is never perfect and learns a probability
distribution :math:`Q` over word sequences which is an approximation of
the true probability distribution :math:`P`. A measure of the quality of
the learned model of the language :math:`L` can be achieved by computing
the cross-entropy of the two probability distributions :math:`P` and :math:`Q`
[Gildea2002]_. Using the notation :math:`\mathbf{\bar{w}}_{n}=(w_{1}\ldots w_{n})`:

.. math::
	\begin{eqnarray}
	H(P,Q) & = & \underset{_{n\rightarrow+\infty}}{\lim}-\frac{1}{n}\sum_{\mathbf{\bar{w}}_{n}\in L}P(\mathbf{\bar{w}}_{n})\log_{2}Q(\mathbf{\bar{w}}_{n})\\
	 & = & \underset{_{n\rightarrow+\infty}}{\lim}-\frac{1}{n}\left(\sum_{\mathbf{\bar{w}}_{n}\in L}P(\mathbf{\bar{w}}_{n})\log_{2}P(\mathbf{\bar{w}}_{n})+\sum_{\mathbf{\bar{w}}_{n}\in L}P(\mathbf{\bar{w}}_{n})\log_{2}\frac{P(\mathbf{\bar{w}}_{n})}{Q(\mathbf{\bar{w}}_{n})}\right)\nonumber \\
	 & = & \begin{array}{ccc}
	H(P) & + & D_{KL}(P||Q),\end{array}
	\end{eqnarray}
where :math:`H(P)` and :math:`D_{KL}(P||Q)` are respectively the entropy rate
and and Kullback-Leibler divergence. Since the Kullback-Leibler divergence
is always positive, the cross-entropy is an upper bound on the entropy
rate of the language, i.e. :math:`H(P)\leq H(P,Q)`, and a language model
with a low cross-entropy (that is is closer to the real entropy) will
supposedly be more accurate.

In practice, the cross entropy is used as an approximation of the
real entropy rate of the language and, assuming that the language
is a stationary and ergodic process, the Shannon-McMillan-Breiman
theorem allows to simplify its expression:

.. math::
	\begin{eqnarray}
	H(P,Q) & = & \underset{_{n\rightarrow+\infty}}{\lim}-\frac{1}{n}\log_{2}Q(w_{1}w_{2}\ldots w_{n})\\
	 & \backsimeq & -\frac{1}{N}\log_{2}Q(w_{1}w_{2}\ldots w_{N}).
	\end{eqnarray}
Hence, the quality of the language model can be estimated from a single
long sequence of :math:`N` words. This leads to the perplexity measure
that is very often used to assess the performance of a language:

.. math::
	\begin{eqnarray}
	Perplexity & = & 2^{H(P,Q)}\\
	 & = & Q(w_{1}\ldots w_{N})^{-\frac{1}{N}}\\
	 & = & \sqrt{\prod_{i=1}^{N}\frac{1}{Q(w_{i}|w_{1}\ldots w_{i-1})}}.
	\end{eqnarray}


Because the entropy rate :math:`H(Q)` measures the average number of information
bit per word according to the model, the perplexity can be thought
of as the average number of possible next word given a context according
to the model. Interestingly, the actual per-letter entropy rate of
the English language was estimated in [Shannon1951]_ using a
Shannon guessing game. The entropy rate reported is of 1.3 bits (for
27 characters).

Another measure of the accuracy of a probabilistic language model
is the average cosine similarity. This measure can be used for simple
language models when the real probability distribution over the sequences
is known. In a Shannon game setting, it simply consists in taking
the normalize inner product between the vector of the real probability
distribution of the next word :math:`\mathbf{p}` and the vector of the
approximated probability distribution of the next word :math:`\mathbf{q}`:

.. math::
	Cos(\mathbf{p},\mathbf{q})=\frac{\langle\mathbf{p},\mathbf{q}\rangle}{\|\mathbf{p}\|_{2}\|\mathbf{q}\|_{2}}.

References
----------

.. [Chen1996] plop
.. [Mnih08ascalable]
.. [AndrewsThesis] plop
.. [Lukosevicius2007] plop
.. [Blei2003] plop
.. [Andrews2009] plop
.. [Boyd-graber2009] plop
.. [Gildea2002] plop
.. [Shannon1951] plop
.. [Turian2010] plop
