Mixture of Experts
==================

Description
-----------

The Mixture of Experts \cite{Bishop2003,Jacobs1991} is a well known
model in which the input data is first partitioned into soft clusters,
each corresponding to an "expert domain", before being processed
by specialized sub-models. Each expert can be a linear regression,
a multinomial logit or any simple model for classification or regression. 


Model definition
----------------

In the mixture of Experts model the conditional probability :math:`P(\mathbf{y}|\mathbf{x})`
of generating :math:`\mathbf{y}` is expressed as a mixture of expert densities
 :math:`P(\mathbf{y}|\mathbf{x},z)`.The mixing coefficients :math:`P(z|\mathbf{x})`
are multinomial probabilities depending on the input data :math:`\mathbf{x}`.
The total probability has the form: 

.. math::
	\begin{eqnarray}
	P(\mathbf{y}|\mathbf{x}) & = & \sum_{z}P(z|\mathbf{x})P(\mathbf{y}|\mathbf{x},z)\\
	 & = & \sum_{z}P(\mathbf{y},z|\mathbf{x}).
	\end{eqnarray}
The marginalization over the latent variable :math:`z` is used to construct
a complex (multi-modal) distribution :math:`P(\mathbf{y}|\mathbf{x})` by
combining simpler (unimodal) expert distributions :math:`P(\mathbf{y}|\mathbf{x},\mathbf{z})`.

FIGURE


The term :math:`P(z|\mathbf{x})` is called a gating model. Its task is
to decide which expert is going to be applied to the prediction of
 :math:`\mathbf{y}` according to :math:`\mathbf{x}`.The gating part of the model
constructs a soft partition of the input space and assigns different
experts :math:`P(\mathbf{y}|\mathbf{x},z)` to different regions. For example,
when words are input to the echo state network, the reservoir constructs
a representation of the history of words in its activation space.
The role of the gates in that case will be to cluster the reservoir
activation space and assign a different expert predictor to each region.
This clustering is soft in the sense that each region is assigned
every expert with different probabilities.

The multinomial mixing coefficients of the gate model are parameterized
by a multinomial logit function:

.. math::
	\begin{equation}
	P(z=i|\mathbf{x})=\frac{\exp(\mathbf{\mathbf{\Theta}_{g,i}^{T}\mathbf{x}})}{\sum_{k}\exp(\mathbf{\mathbf{\Theta}_{g,i}^{T}\mathbf{x}})},\label{eq:gateeq}
	\end{equation}
where :math:`\mathbf{\Theta}_{g,i}` are parameter vectors. 

Here we choose to restrict the form of the expert densities to linear
regressions or multivariate logit. The expert distribution can be
parameterized by:

.. math::
	\begin{eqnarray}
	P(\mathbf{y}|\mathbf{x},z=i) & = & \frac{1}{(2\pi)^{D/2}\sigma^{D}}\exp(-\frac{1}{2\sigma^{2}}(\mathbf{y-\mathbf{\mathbf{\Theta}_{e,i}^{T}\mathbf{x}})^{T}(\mathbf{y-\mathbf{\Theta}_{e,i}^{T}\mathbf{x}))}},
	\end{eqnarray}
where :math:`\mathbf{\Theta}_{e,i}` is a matrix, in the case of a regression
task, or 

.. math::
	\begin{equation}
	P(\mathbf{y}|\mathbf{x},z=i)=\frac{\exp(\mathbf{\mathbf{\Theta}_{e,i}^{T}\mathbf{x}})}{\sum_{k}\exp(\mathbf{\mathbf{\Theta}_{e,k}^{T}\mathbf{x}})},
	\end{equation}
where :math:`\mathbf{\Theta}_{e,i}` is a vector, for a classification task.

FIGURE


Since, the activation of the reservoir :math:`\mathbf{x}` is a representation
of the past history f a sentence, it is expected that the gating model
will be able to extract interesting linguistic features. Since the
features introduced in the previous chapter are intended to re-organize
the activation of the recurrent layer, the gating model is expected
to give a higher level information on the kind of representation developed.
Those features will be extracted automatically during the training
phase to maximize the likelihood of the data.

Learning
--------

Learning is done using Expectation-Maximization as described in REF.

Code
----------

.. automodule:: esnlm.readouts.moe
   :members:
   :undoc-members: