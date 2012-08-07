Markovian organization
======================

Elman's grammar
---------------

In order to assess the performance of the proposed algorithms, we
want to compare the results we obtain to the performance of other
neural network language models. Although recent advances in language
modeling have been made using sophisticated feed-forward networks
\cite{Bengio:2003:NPL:944919.944966,collobert:2008,Mnih08ascalable,Mnih2007},
we restricted our comparison to simple recurrent neural networks and
conventional echo state networks. 

Work on simple recurrent neural networks applied to a language modeling
task dates back to Elman's experiment in 1991 \cite{Elman:1991:DRS:125342.125347}.
Jeffrey Elman trained recurrent neural networks on a simple artificial
grammar whose statistical and syntactic properties were readily available.
This artificial grammar is quite simple. Nevertheless it exhibits
several interesting behavior and can produce complex sentence with
center embedding. This grammar is today often referred to as Elman's
grammar. 

GRAMMAR

A comparison between echo state networks and simple recurrent network
was studied in \cite{Tong:2007:SIL:1265618.1265889}. According to
results obtained in that experiment, echo state network show performances
similar to simple recurrent network on a simple artificial grammar.
On one hand the advantage of echo states networks over simple recurrent
networks was that no special trick was used during the learning phase.
In fact, the one shot learning procedure is fast in comparison to
the gradient descent algorithm usually used to train recurrent neural
network. On the other hand, at a similar level of performance, the
number of unit in the reservoir layer was far greater than in a simple
recurrent network. This questions the ability of echo states network
to be applied to more realistic natural language processing tasks
where the vocabulary and the number of sentence structures are increased
to describe the everyday English language. 

In the next section we describe how an echo state network can be used
as a language model.


A Shannon Game
--------------

The task performed by the echo state network is a basic Shannon guessing
game where the network is asked to provide the probability of the
next word in a sentence given the past words. Unlike :math:`\textit{n}`-gram
or feed-forward neural models, the context information available is
not of fixed length: the recurrent layer can in theory store information
about an arbitrarily long context. 

More formally, the task is to model :math:`P(w_{t}|w_{0}...w_{t-1})` where
:math:`w_{t}` is the next word and :math:`w_{0}...w_{t-1}` is the sequence of
the previously input words, i.e. its context. Each word :math:`v`
in a vocabulary :math:`V` of size :math:`|V|` is mapped to a unique binary vector
:math:`\mathbf{u}` with only one nonzero component. A sentence :math:`w_{1}w_{2}...w_{l}`
of length :math:`l` is thus transformed into a sequence :math:`\mathbf{u}_{0},\mathbf{u}_{1},...,\mathbf{u}_{l}`
of binary vectors. At each time :math:`t`, the binary vector :math:`\mathbf{u}_{t}`
is fed to the recurrent layer. The probability of each candidate word
in the vocabulary :math:`V` is then computed by the readout function according
to the activation :math:`E(\mathbf{u}_{0},...,\mathbf{u}_{t})` of the recurrent
layer. The final probability vector :math:`\mathbf{p}=[p_{k}]_{k=1,..,|V|}\in\mathbb{R}^{|V|}`
contains the probability of every possible candidate :math:`v\in V` along
its components.

.. math::
	\begin{align}
	\begin{split}\mathbf{p}_{t} & =[P(v_{k}|w_{1}...w_{t})]_{k=1,..,|V|}\\
	 & =h(E(\mathbf{u}_{0},...,\mathbf{u}_{t}))\\
	 & =h(g(\mathbf{u}_{t}),\mathbf{x}_{t-1}),
	\end{split}
	\label{eqprob}
	\end{align}

where :math:`v_{k}` is the :math:`k^{th}` word in the vocabulary :math:`V`.

In equation (\ref{eqprob}) the readout :math:`h` is a linear function.
Since the probability vector :math:`\mathbf{p}` might contain negative
components and not sum to one, it is rescaled appropriately. 

This language model can then assign a probability to every sequence
in the following way:

.. math::
	\begin{eqnarray}
	p(w_{1}\ldots w_{n}) & = & \prod_{n=1}^{n}P(w_{n}|w_{1}w_{2}w_{3}\ldots w_{n-1})\\
	 & = & \prod_{t=1}^{n}[h(g(\mathbf{u}_{t}),\mathbf{x}_{t-1})]_{w_{t}}.
	\end{eqnarray}



The Markovian organization of the state-space
---------------------------------------------

The echo state property of the network gives an intuitive justification
to why applying an echo state network to a language task may yield
interesting results even in the absence of learning in the recurrent
layer. Scaling the weights of the reservoir to obtain the echo state
property ensures that the activity of the recurrent layer driven by
input sequences with similar history will converge to close regions
of the state space. The distance between two state-space activations
:math:`E(\mathbf{u}_{0}\mathbf{\bar{u}}_{1}^{t})` and :math:`E(\mathbf{v}_{0}\bar{\mathbf{u}}_{1}^{t})`
sharing the same history of input :math:`\bar{\mathbf{u}}_{1}^{t}=\mathbf{u}_{1}\mathbf{u}_{2}\ldots\mathbf{u}_{t}`
but differing only by their first terms :math:`\mathbf{u}_{0}\not=\mathbf{v}_{0}`
can be expressed by:

.. math::
	\begin{eqnarray}
	d & \equiv & d(E(\mathbf{u}_{0}\mathbf{\bar{u}}_{1}^{t}),E(\mathbf{v}_{0}\bar{\mathbf{u}}_{1}^{t}))\\
	 & = & d(g(\mathbf{W}^{in}\mathbf{u}_{t}+WE(\mathbf{u}_{0}\mathbf{\bar{u}}_{1}^{t-1})),g(\mathbf{W}^{in}u_{t}+\mathbf{W}E(\mathbf{v}_{0}\mathbf{\bar{u}}_{1}^{t-1})))\\
	 & \leq & d(\mathbf{W}^{in}u_{t}+\mathbf{W}E(\mathbf{u_{0}}\mathbf{\bar{u}}_{1}^{t-1}),\mathbf{W}^{in}u_{t}+\mathbf{W}E(\mathbf{v}_{0}\mathbf{\bar{u}}_{1}^{t-1}))\\
	 & = & d(\mathbf{W}E(\mathbf{u}_{0}\mathbf{\bar{u}}_{1}^{t-1}),\mathbf{W}E(\mathbf{v}_{0}\mathbf{\bar{u}}_{1}^{t-1}))\\
	 & \leq & \bar{\sigma}(\mathbf{W})d(E(\mathbf{u}_{0}\mathbf{\bar{u}}_{1}^{t-1}),E(\mathbf{v}_{0}\mathbf{\bar{u}}_{1}^{t-1})),
	\end{eqnarray}
	
where :math:`\textit{d}` is the Euclidean distance, and :math:`\bar{\sigma}(\mathbf{W})`
is the largest singular value of :math:`\mathbf{W}`. By repeating the same
this calculation :math:`t` times, we see that the distance between the
state space representation of two input sequences differing only by
their first term verifies the inequality:

.. math::
	\begin{equation}
	d(E(\mathbf{u}_{0}\mathbf{\bar{u}}_{1}^{t}),E(\mathbf{v}_{0}\mathbf{\bar{u}}_{1}^{t})\leq\bar{\sigma}(\mathbf{W})^{t}d(E(\mathbf{u}_{0}),E(\mathbf{v}_{0})).
	\end{equation}


Assuming :math:`\bar{\sigma}(W)<1`, which a sufficient condition on the
echo state property, it is then possible to cluster the state space
into regions corresponding to the :math:`\textit{n}`-gram suffix of the
sequence. Moreover the clusters will tend to have a fractal organization.
For example it is possible to find a region of the state space corresponding
to the suffix :math:`(...,\mathbf{u}_{t-3},\mathbf{u}_{t-2},\mathbf{u}_{t-1},\mathbf{u}_{t})`
in a ball of radius :math:`\bar{\sigma}^{4}D` which will itself be a sub-region
of the :math:`(...,\mathbf{u}_{t-2},\mathbf{u}_{t-1},\mathbf{u}_{t})` region
which has radius :math:`\bar{\sigma}^{3}D` (where :math:`D` can be chosen to
be the radius of a ball containing all the trajectories of the recurrent
layer). In the more general case where :math:`\bar{\sigma}(W)<1` does not
necessarily hold, the uniformly state contracting property of echo
state networks \cite{Jaeger2001a} implies a similar organization
of the reservoir activation. 

These clusters can be visualized in figure \ref{fig:1-gram-regions}
and figure \ref{fig:2-gram-regions-for}, respectively corresponding
to the 1-gram and 2-gram clustering of the reservoir activation of
an echo state network processing sentences in Elman's grammar.

FIGURE


This observation leads to a better understanding of the method used
by echo states networks to model sentences (and more generally to
perform any Markov-like task): echo state networks correspond loosely
to variable length Markov models \cite{Tino02markovianarchitectural}.

In summary, the fractal or Markovian representation developed in the
reservoir layer is linked to :math:`n`-gram models. This leads to a reflexion
on the usefulness of echo state networks for language modeling. Echo
state network may be more flexible than :math:`n`-gram models because of
their ability to represent increasingly long history in their state
space. However, because of the fractal property of the representation,
it can be expected that a very precise readout is necessary to extract
information about words that have been seen in a "not so recent"
past. In that way, echo state networks share some of the :math:`n`-gram
models inability to model long term dependencies. In addition, since
the readout is linear, the probability produced by the network in
its last layer may be a poor approximation and requires a huge number
of reservoir units.

This thesis tries to address the problems discussed in this section
by leveraging the representation constructed by the reservoir with
a pre-recurrent processing of the input. Also, better ways to extract
more fine grained information from the reservoir are investigated.
At the same time, efforts are made to keep the learning procedures
fast and easy.

The next chapters present in more detail the modification made to
the basic echo state network architecture.

