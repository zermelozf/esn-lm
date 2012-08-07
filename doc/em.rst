Expectation-Maximization
========================


Description
-----------

The parameters :math:`\Theta` of the model are learned by maximizing the
likelihood of the data, 

.. math::
	\begin{equation}
	\Theta^{*}=\underset{\Theta}{\arg\max}\ \mathcal{L}(\mathcal{X},\Theta),
	\end{equation}

or equivalently, the log-likelihood of the data since the logarithm
function is strictly increasing. Here the samples :math:`(\mathbf{x}_{n},\mathbf{y}_{n})\in\mathcal{X}`
are assumed to be independent \footnote{This is not the case for Echo State Networks but we make this simplification
nevertheless in order to decouple the samples and make the training
easier.
}:

.. math::
	\begin{eqnarray}
	\text{\ell}(\mathcal{X},\Theta) & = & \ln\mathcal{L}(\mathcal{X},\Theta)\\
	 & = & \ln\sum_{z=1}^{K}p(z|\mathbf{x}_{n})p(\mathbf{y}_{n},z|\mathbf{x}_{n}).
	\end{eqnarray}


Although the gating and experts models \footnote{In order to keep the notations uncluttered we often omit to specify
explicitly the dependency of the different probability distribution
on their parameters.} :math:`p(z|\mathbf{x})` and :math:`p(\mathbf{y},z|\mathbf{x})` can be learned
easily when they are used as standalone models (by learning a simple
linear regression model or a logit regression), the summation inside
the logarithm that appears in the Mixture of Experts model renders
the learning of the parameters more complex. However, using the Maximization-Expectation
algorithm allows to express a lower bound on the log-likelihood in
which the different sub-model can be estimated independently.

Using Jensen's inequality the log-likelihood can be rewritten as:

.. math::
	\begin{eqnarray}
	\text{\ell}(\mathcal{X},\Theta) & = & \sum_{n=1}^{M}\ln\sum_{z=1}^{K}p(\mathbf{y}_{n},z|\mathbf{x}_{n})\\
	 & \geq & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln\frac{p(\mathbf{y}_{n},z|\mathbf{x}_{n})}{q_{n}(z)}\\
	 & = & \mathcal{F}(\mathcal{X},q,\Theta),
	\end{eqnarray}

where :math:`q_{n}\geq0` satisfies :math:`\sum_{z=1}^{K}q_{n}(z)=1`. In this
new form the summation over the latent variable :math:`z` has been taken
out of the logarithm. Moreover we define 

.. math::
	\begin{eqnarray}
	\mathcal{Q}(\mathcal{X},q,\Theta) & = & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln p(\mathbf{y}_{n},z|\mathbf{x}_{n})\\
	H(\mathcal{X},q) & = & -\sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln q_{n}(z),
	\end{eqnarray}

such that 

.. math::
	\begin{equation}
	\mathcal{F}(\mathcal{X},q,\Theta)=\mathcal{Q}(\mathcal{X},q,\Theta)+H(\mathcal{X},q).
	\end{equation}


* 	In the Expectation step, this lower bound on the log-likelihood is
	mized with respect to :math:`q_{nz}`, leading to:
	
	.. math::
		\begin{equation}
		q_{n}(z)=p(z|\mathbf{x},\mathbf{y}).
		\end{equation}

	The variable :math:`q_{n}(z)` thus corresponds to the posterior probability
	of :math:`z` given :math:`\mathbf{x}` and :math:`\mathbf{z}`.

* 	In the Maximization step, :math:`\mathcal{F}(\mathcal{X},q,\Theta)` is
	maximized with respect to the parameters of the model. Since :math:`q_{n}(z)`
	does not depend on the parameters :math:`\Theta` of the model, the objective
	function can be simplified during this step:


	.. math::
		\begin{eqnarray}
		\Theta^{*} & = & \underset{\Theta}{\arg\max}\ \mathcal{B}(\mathcal{X},q,\Theta)\\
		 & = & \underset{\Theta}{\arg\max}\ \mathcal{Q}(\mathcal{X},\Theta)+H(q)\\
		 & = & \underset{\Theta}{\arg\max}\ \mathcal{Q}(\mathcal{X},\Theta).
		\end{eqnarray}


In a sense, the EM algorithm transforms a learning problem with unobserved
variable (logarithm of a summation in :math:`\text{\ensuremath{\ell}}(\mathcal{X},\Theta)`)
into a new problem where all the variables are observed (summation
of a logarithm in :math:`\mathcal{F}(\mathcal{X},\Theta)`). The new problem
is the weighted average of all the possible realizations of the former
problem and the weights are the posterior probability of those realizations.

Independent estimation
----------------------

Let us now explicit the form of :math:`\mathcal{Q}(\mathcal{X},\Theta)`
for linear and multinomial logit experts: 

.. math::
	\begin{eqnarray}
	\mathcal{Q}(\mathcal{X},\Theta) & = & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln p(\mathbf{y}_{n},z|\mathbf{x}_{n})\\
	 & = & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln p(z|\mathbf{x}_{n})+\sum_{z=1}^{K}\left(\sum_{n=1}^{M}q_{n}(z)\ln p(\mathbf{y}_{n}|\mathbf{x}_{n},z)\right)\nonumber \\
	 & = & \mathcal{Q}_{g}(\mathcal{X},\Theta_{g})+\sum_{z}\mathcal{Q}_{e,z}(\mathcal{X},\Theta_{e,z}),
	\end{eqnarray}

where 

.. math::
	\begin{eqnarray}
	\mathcal{Q}_{g}(\mathcal{X},\Theta_{g}) & = & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln p(z|\mathbf{x}_{n})\\
	\mathcal{Q}_{e,z}(\mathcal{X},\Theta_{e,z}) & = & \sum_{n=1}^{M}q_{n}(z)\ln p(\mathbf{y}_{n}|\mathbf{x}_{n},z)
	\end{eqnarray}

are the objective functions for the gating and expert models respectively.
The parameters in each of the sub-objectives have been decoupled hence
the sub-models can be trained independently:

.. math::
	\begin{eqnarray}
	\Theta_{g}^{*} & = & \underset{\Theta_{g}}{\arg\max}\ \mathcal{Q}_{g}(\mathcal{X},\Theta_{g})\label{eq:gateoptfunc}\\
	\Theta_{e,z}^{*} & = & \underset{\Theta_{e,z}}{\arg\max}\ \mathcal{Q}_{e,z}(\mathcal{X},\Theta_{e,z}).
	\end{eqnarray}

:math:`\Theta_{g}^{*}` can be found by minimizing the cross-entropy between
:math:`q_{n}(z)` and :math:`p(z|\mathbf{x})` while the :math:`\Theta_{e,z}^{*}` can
be found by solving a weighted least square problem in the regression
case or by minimizing a cross-entropy function in the classification
case.

Code
----

.. automodule:: esnlm.optimization.em
   :members:
   :undoc-members:


   