Linear Regression
=================

Description
-----------

The echo state network is a sequence model mapping an input sequence 
:math:`\mathbf{\bar{u}}` to an output sequence :math:`\bar{\mathbf{y}}` . 
In traditional recurrent neural networks, the parameter update is usually
computed by applying the chain rule to the network unrolled in time
and making a small step in the parameter space in the direction of
the gradient of the loss function with respect to the parameters of
the model. This procedure can be computationally quite demanding,
requires choosing an appropriate learning rate to ensure convergence
in the parameter space and cannot be applied (exactly) to sequences
of infinite length. In echo state networks however, since all the
parameters to be learned are located after the recurrent layer, the
data can essentially be viewed as a set of training pairs  :math:`\mathcal{X}=\{(\mathbf{x}_{n},\mathbf{y}_{n})\backslash n=1\ldots M\}`
where  :math:`\mathbf{x}_{n}` and :math:`\mathbf{y}_{n}` are respectively the
reservoir activation and output value at time :math:`n`. Any supervised
algorithm can then be used to learn a mapping between :math:`\mathbf{x}`
and :math:`\mathbf{y}`. 

In practice, it often assumed that the reservoir is able to construct
a useful nonlinear transformation of the input data and that the task
of the readout is merely to extract information from this representation.
The extraction of nonlinear feature being achieved by the reservoir,
the most common readout function :math:`h` is a linear transformation of
the reservoir activation and the loss :math:`L` is the squared error loss:

.. math::

	\begin{eqnarray}
	h(\mathbf{x}_{n}) & = & \mathbf{W}^{out}\mathbf{x}_{n},\\
	L(\mathbf{Y},h(\mathbf{X})) & = & \sum_{(\mathbf{x}_{n},\mathbf{y}_{n})}(\mathbf{y}_{n}-h(\mathbf{x}_{n}))^{2},
	\end{eqnarray}
	
where :math:`\mathbf{X}\in\mathbb{R}^{M\times D},\mathbf{Y}\in\mathbb{R}^{M\times O}`
are matrices with row :math:`n` corresponding respectively to :math:`\mathbf{x}_{n}`
and :math:`\mathbf{y}_{n}`. 

From a statistical point of view \cite{Hastie2009}, if :math:`(\mathbf{x}_{n},\mathbf{y}_{n})`
are samples from a probability distribution :math:`P(X,Y)`, the loss function :math:`L` is proportional to the empirical estimator of the expected mean
square error:

.. math::
	\begin{eqnarray}
	\mathbb{E}[(Y-h(X))^{2}] & = & \sum_{(\mathbf{x},\mathbf{y})}(\mathbf{y}-h(\mathbf{x}))^{2}p(\mathbf{x},\mathbf{y}).
	\end{eqnarray}


Moreover, finding :math:`h` by minimizing the expectation of the squared
error leads to a solution where :math:`h(\mathbf{x})` is the expectation
of :math:`Y` given :math:`X=\mathbf{x}`. 

.. math::
	\begin{eqnarray}
	h(\mathbf{x}) & = & \underset{c}{\arg\min}\ \mathbb{E}_{Y|X}[(Y-c)^{2}|X=\mathbf{x}]\\
	 & = & \mathbb{E}_{Y|X}[Y|X=\mathbf{x}].\label{eq:YcondX}
	\end{eqnarray}


This result is useful to understand what is learned by the model when
it is trained by minimizing the squared error function. In particular,
in \ref{sec:A-Shannon-Game} the network is trained to predict the
probability of the next word :math:`[p(word_{i})]_{i=1,...,|V|}` for every
word in the vocabulary :math:`V` when it is only given a one-hot representation
of that word :math:`\mathbf{y}_{n}=[\delta_{i,word}]_{i=1,..,|V|}`. This
is possible since :math:`\mathbb{E}(Y)=[p(word_{i})]_{i=1,...,|V|}`.

It is to be noted that equation (\ref{eq:YcondX}) is valid only if
the form of the readout function is not constrained. If the readout
is linear, the prediction might not even be a real probability distribution.

Learning the parameters by minimizing the loss function in the linear
readout case is relatively easy: 

.. math::
	\begin{eqnarray}
	\mathbf{W}^{out} & = & \underset{\mathbf{W}}{\arg\max}L(\mathbf{Y},h(\mathbf{X}))\\
	 & = & \underset{\mathbf{W}}{\arg\max}(\mathbf{Y}-\mathbf{\mathbf{W}X}))^{T}(\mathbf{Y}-\mathbf{W}\mathbf{X})\\
	 & = & \mathbf{Y}{}^{T}\mathbf{X}(\mathbf{X}\mathbf{X}^{T})^{-1}.
	\end{eqnarray}


The linear regression is the conventional readout for echo state networks.
It can be learned in one shot using least square fitting. 
There is no intercept by default so it must be added "manually".
Since the prediction of a linear function is not well normalized and may even be 
less than 0, It is renormalized appropriately.

References
----------

.. automodule:: esnlm.readouts.linear
   :members:
   :undoc-members: