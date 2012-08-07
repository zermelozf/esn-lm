Echo State Networks
===================

Echo State Networks (ESNs) \cite{Jaeger2001a} are a rather recent
development in the field of recurrent neural networks and have gained
popularity amongst researchers for their good performances on tasks
such as time series prediction or speech recognition as well as for
their easy implementation and training.


Model definition
----------------

An Echo State Network is a simple, discrete-time recurrent neural
network with three layers: 

* an input layer :math:`\mathbf{u}\in\mathbb{R}^{I}` , 
* a recurrent layer :math:`\mathbf{x}\in\mathbb{R}^{D}`  also called reservoir,
* an output layer :math:`\mathbf{y}\in\mathbb{R}^{O}`  extracting information
	m the reservoir. 

The core component of an Echo State Network is the reservoir: a large
collection of randomly connected units, each of which computes a nonlinear
transformation of the input signal. A readout function is then used
to extract the information contained in the representation of the
input sequence constructed by the reservoir.

The echo state network is fed at every time step :math:`t`  with input vector
:math:`\mathbf{u}_{t}`  which drives the dynamic of recurrent layer :math:`\mathbf{x}_{t}` 
and outputs vectors :math:`\mathbf{y}_{t}`. The input is propagated through
the layers according to the system of equations:

.. math::
	\begin{eqnarray}
	\mathbf{x}_{t} & = & E(\mathbf{u}_{0},...,\mathbf{u}_{t})\\
	 & = & g(\mathbf{W}^{in}\mathbf{u}_{t}+\mathbf{W}\mathbf{x}_{t-1})\\
	\mathbf{y}_{t} & = & h(\mathbf{x}_{t}),\\
	 & = & \mathbf{W}^{out}\mathbf{x}_{t}.
	\end{eqnarray}

Throughout this document the following notations will be used:

* 	:math:`g`  and :math:`h`  are respectively the activation of the recurrent layer
	e readout function. In the conventional ESN, :math:`g=\tanh`  and
	linear as in equation (\ref{eq:weightmat}). 
* 	:math:`\mathbf{W}^{in}\in\mathbb{R}^{D\times I}`  are the input weights.
	The input weights are usually drawn randomly from :math:`\{-1,0,1\}`  and
	can be rescaled in order to position the reservoir response to the
	input appropriately on the nonlinearity. This allows to avoid the
	reservoir response to be close to binary values when the input dimensionality
	is high, or may be useful to test the reservoir in near linear mode
	when the input scaling is very small.
* 	:math:`\mathbf{W}\in\mathbb{R}^{D\times D}`  is the matrix containing the
	reservoir weights. Roughly speaking, the particularity of ESNs is
	that this matrix is randomly drawn and kept fixed during the experiment.
	The properties of :math:`\mathbf{W}`  are discussed in more details in \ref{sub:The-Echo-State}.
* 	:math:`\mathbf{W}^{out}\in\mathbb{R}^{O\times D}`  is the output matrix.
	` \mathbf{W}^{out}`  is usually the only set of parameters that is
	learned in a conventional ESN. Learning the output matrix is described
	in \ref{sub:Learning-with-Least}.
* 	:math:`E`  is the echo state function. It is to be noted that the echo state
	function has a growing number of input variable. :math:`E(\mathbf{u}_{0},...,\mathbf{u}_{t})` 
	is actually an alternative way of expressing :math:`\mathbf{x}_{t}`  (the
	activation of the reservoir at time :math:`t`  ) making explicit the dependency
	on the input sequence :math:`(\mathbf{u}_{0},...,\mathbf{u}_{t})`.


In order to keep the complexity of the system minimal, we chose not
to admit back-propagation of the output units to the recurrent layer
nor direct connection of the input units to the output layer. In a
sense, this allows the network to be closer to the well known feed-forward
architecture, a characteristic that is used later.

FIGURE

In the same way feed-forward neural network are useful to learn a
mapping between two vectors, recurrent neural network are used to
learn a mapping between sequences. When feed-forward neural networks
can learn simple function, recurrent neural network (and thus echo
state networks) are able to learn dynamical systems. In fact, given
only some production of an existing real dynamical system, it is hoped
that the recurrent neural network will be able to reconstruct in its
recurrent layer the state space of this dynamical system and allow
a deeper understanding of the nature of the dynamical system. This
``hope'' is supported by Takens' theorem \cite{Noakes1991} which
describes the conditions under which a chaotic dynamical system can
be reconstructed from a sequence of observations. This theorem is
also useful to justify the use of time-delay neural networks.


The Echo State Property
-----------------------

To this point, it may seem that the main difference between a simple
recurrent neural network \cite{Elman:1991:DRS:125342.125347} and
an echo state network is the fact that :math:`\mathbf{W}` , the weight matrix
of the reservoir, is fixed. Although this is mainly true from a training
point of view (only the output weights are to be learned in the ESN
case), fixing the weights of a simple recurrent neural network does
not automatically yield an echo state network. The defining characteristic
of an echo state network is that it exhibits the echo state property!
Obviously the previous statement is dangerously close to a tautology.
Let's refer to \cite{Jaeger2001a} to define more formally the echo
state property.

\begin{definition} Assume standard compactness. The network has :math:`echo\ states` , if the network state :math:`\mathbf{x}_{n}`  is uniquely determined by any left-infinite input sequence :math:`\bar{u}^{-\infty}` . More precisely, this means that for every input sequence :math:`(..., \mathbf{u}_{n-1}, \mathbf{u}_{n}) \in U^{-\mathbb{N}}` , for all state sequence :math:`(..., \mathbf{x}_{n-1}, \mathbf{x}_{n})`  and  :math:`(..., \mathbf{x'}_{n-1}, \mathbf{x'}_{n}) \in A^{-\mathbb{N}}` , where :math:`\mathbf{x}_{i}=g(\mathbf{\mathbf{x}_{i-1}}, \mathbf{u}_{i})`  and :math:`\mathbf{x'}_{i}=g(\mathbf{\mathbf{x'}_{i-1}}, \mathbf{u}_{i})` , it holds that :math:`\mathbf{x}_{n}=\mathbf{x'}_{n}` .\end{definition}

Unfortunately, this formal definition does not allow an intuitive
characterization of echo state networks. However, it can be shown
\cite{Jaeger2001a} that the echo state property is equivalent to
the network being \emph{state contracting}, \emph{state forgetting}
or \emph{input forgetting. }We will not rigorously define these different
properties but rather try to give an intuitive explanation of what
the echo state property is. To this end we introduce the state update
operator :math:`T:(\mathbf{x}_{n},\mathbf{\bar{u}}_{h})\mapsto\mathbf{x}_{n+h}` 
where :math:`\mathbf{\bar{u}}_{h}=\mathbf{u}_{1},\ldots,\mathbf{u}_{h}` .

1. 	A network has the echo state property if the dynamical system it describes
	is contracting, that is if two trajectories starting from different
	initial points in the reservoir state-space end up in similar region
	of the state space when the system is driven by the same input sequence
	for a "sufficiently long time".
#. 	An echo state network is state forgetting for similar reasons. Independently
	of the initial state, the current activation of the reservoir will
	depend only on the .history of the input sequence given that a sufficiently
	long history is taken into account. This property is the same as the
	contracting property for left-infinite sequences.
#. 	An echo state network is input forgetting since independently of the
	"far" past input sequence, the activation of the reservoir is
	determined only by the "recent" history of the network.


The echo state property and the three equivalent properties described
above can be stated as: 

.. math::
	\begin{equation}
	\forall\bar{\mathbf{u}}=(\mathbf{u}_{0},\mathbf{u}_{1},\ldots,\mathbf{u}_{h}),\|T(\mathbf{x}_{n},\mathbf{\bar{u}}_{h})-T(\mathbf{x}'_{n},\mathbf{\bar{u}}_{h})\|\longrightarrow0,
	\end{equation}

when :math:`h\longrightarrow+\infty` .

In practice, a sufficient and a necessary condition can be used to
ensure that a network exhibits the echo state properties.

*Lemma*: A sufficient condition on the reservoir weight matrix :math:`\mathbf{W}`  ensuring that the network exhibits the echo state property is that its largest singular value :math:`\bar{\sigma}(\mathbf{W})`  should be less than 1. 


.. math::
	\bar{\sigma}(\mathbf{W}) < 1

	

*Lemma*: A necessary condition on the reservoir weight matrix :math:`\mathbf{W}`  ensuring that the network exhibits the echo state property is that its spectral radius :math:`\bar{\rho}(\mathbf{W})` , i.e. its largest eigenvalue, should be less than 1. 


.. math::
	\bar{\rho}(\mathbf{W}) < 1

	

These bound on the echo state property are not tight. A less strict
sufficient condition on the matrix :math:`\mathbf{W}`  can be found in \cite{Buehner_Young_2006}.
Instead of the matrix norm induced by the Euclidean norm :math:`\|\mathbf{W}\|_{2}=\underset{\mathbf{x}\neq\mathbf{0}}{\sup}\frac{\|\mathbf{Wx}\|_{2}}{\|\mathbf{x}\|_{2}}=\bar{\sigma}(\mathbf{W})` ,
the induced D-norm :math:`\|\mathbf{W}\|_{\mathbf{D}}=\bar{\sigma}(\mathbf{D}\mathbf{W}\mathbf{D}^{-1})` 
is used with the same condition :math:`\|\mathbf{W}\|_{\mathbf{D}}<1`.
Since in a finite vectorial space all norms are equivalent, the same
result on the echo state property can be derived from this norm though
with a less restrictive condition by choosing an appropriate matrix
:math:`\mathbf{D}`.

Although it does not always ensure the echo state property, the necessary
condition on the reservoir weight matrix, i.e. :math:`\rho(\mathbf{W})<1` ,
is commonly used as a good heuristic to construct the reservoir. In
practice, all the weight matrices are usually initialized randomly.
The reservoir matrix is then rescaled so that its spectral radius
be less than 1.

Finally, the echo state property can be linked to the Markovian organization
of the reservoir state-space as will be discussed in \ref{sec:The-Markovian-organization}.
This particular representation , although very useful for a wide variety
of tasks, is the reason for all the modifications of the conventional
echo state network architecture that are studied in this document.

Code
----

The code implementing echo state networks relies on the Oger library __TODO__:REF

See also:
^^^^^^^^^


.. automodule:: esnlm.utils.functions
   :members: sparseReservoirMatrix
