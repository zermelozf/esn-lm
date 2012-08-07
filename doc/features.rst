Features
========

Description
-----------

The system of equations describing the network augmented with a feature
layer is:

.. math::
	\begin{eqnarray}
	\mathbf{x}_{t} & = & E(f(\mathbf{u}_{0}),...,\mathbf{f(u}_{t}))\\
	 & = & g(f(\mathbf{u}_{t}),\mathbf{x}_{t})\\
	\mathbf{y}_{t} & = & h(\mathbf{x}_{t}),
	\end{eqnarray}

where :math:`f` is the function extracting features from the input :math:`\mathbf{u}`.
Following the idea presented in \cite{Bengio:2003:NPL:944919.944966}
and also used in \cite{collobert:2008}, :math:`f` will simply be a linear
projection of the input vector :math:`\mathbf{u}` on a vector space whose
dimensionality :math:`F` can be chosen:

.. math::
	\begin{equation}
	f(\mathbf{u})=\mathbf{W}^{f}\mathbf{u},
	\end{equation}
	
where :math:`\mathbf{W}^{f}\in\mathbb{R}^{I\times F}`. Since the input
vectors :math:`\mathbf{u}` are actually binary with only one nonzero component,
the feature of a word is just the recopy of one column of the feature
matrix into the feature layer. The feature matrix is learned by a
gradient descent optimization which will be explained in \ref{sec:Learning-with-Gradient},
hence the word features will be fixed after convergence.

FIGURE

Learning with gradient descent
------------------------------

The features matrix :math:`\mathbf{W}^{f}` and output matrix :math:`\mathbf{W}^{out}`
are learned by minimizing the cost function :math:`C` of every sequences :math:`s=(\mathbf{u}_{0},...,\mathbf{u}_{T_{s}})` in the training
set \textit{S} . At each time step the prediction of the network is
compared to the actual next word in the sequence,

.. math::
	\begin{equation}
	C=\frac{1}{2}\sum_{s\in S}\sum_{t=1}^{T_{s}-1}(\mathbf{p}_{s,t}-\mathbf{u}_{s,t+1})^{T}(\mathbf{p}_{s,t}-\mathbf{u}_{s,t+1}).\label{eq:costfunc}
	\end{equation}


In the basic architecture, since all the parameters of the models
are after the recurrent layer, it is possible to learn a linear readout
in one shot. Unfortunately, introducing a feature layer causes the
update of the feature matrix to be dependent on several time-steps
and requires back-propagation Through Time of the gradient error.
However the echo state property allows neglecting the terms of the
gradient back-propagated several times through the recurrent layer.
In fact, the echo state property is responsible for the vanishing
gradient effect described in \cite{Bengio_learninglong-term}. Informally,
since the influence of the feature :math:`f(\mathbf{u}_{t})` is attenuated
through time because of the echo state property, it has only a very
limited impact on the state of the recurrent layer :math:`\mathbf{x}_{t+n}`
if :math:`n` is a sufficient time difference. Hence, the update of the
feature representation :math:`f(\mathbf{u}_{t})` conditioned on the prediction
of :math:`\mathbf{u}_{t+1}` can be several orders of magnitude bigger than
the update caused by the prediction of :math:`\mathbf{u}_{t+n}` . 

In order to make the derivation of the learning procedure clearer,
we briefly derive the matrix formulation of back-propagation through
time. We consider that the echo state network fed with a sequence 
\footnote{In this paragraph we use superscript indexes for convenience. This
is to avoid confusion between vector components and vector sequences.
} :math:`\mathbf{\bar{u}}=\mathbf{u}^{1},\ldots,\mathbf{u}^{L}` is unrolled
in time. It has a structure equivalent to a feed-forward network with
shared parameters at every instance :math:`\mathbf{x}^{t}` of the recurrent
layer at time :math:`t`. The system of equations describing the unrolled
echo state network is:

.. math::
	\begin{eqnarray}
	\mathbf{f}^{t} & = & \mathbf{W}^{f}\mathbf{u}^{t}\\
	\mathbf{a}^{t} & = & \mathbf{W}\mathbf{x}^{t-1}+\mathbf{W}^{in}\mathbf{f}^{t},\ for\ 1\leq t<L\\
	\mathbf{x}^{t} & = & g(\mathbf{a}^{t})\\
	\mathbf{y}^{t} & = & \mathbf{W}^{out}\mathbf{x}^{t-1}
	\end{eqnarray}
We are interested in the update of the feature :math:`\mathbf{f}^{1}` caused
by the prediction error :math:`e=(\mathbf{y}^{L-1}-\mathbf{u}^{L})^{T}(\mathbf{y}^{L-1}-\mathbf{u}^{L})`
at time :math:`L-1` that corresponds to only one term in the cost function
\ref{eq:costfunc}. The gradient of :math:`e` with respect to :math:`\mathbf{f}` 

.. math::
	\begin{eqnarray}
	\frac{\partial e}{\partial f_{k}^{f}} & = & \underset{i,j}{\sum}\frac{\partial e}{\partial a_{i}^{1}}\frac{\partial a_{i}^{1}}{\partial f_{k}^{1}}\\
	 & = & \sum_{i}\frac{\partial e}{\partial a_{i}^{1}}W_{i,k}^{in},
	\end{eqnarray}
can be expressed in matrix form:

.. math::	
	\begin{equation}
	\nabla_{\mathbf{f}}e=(\mathbf{W}^{in})^{T}\mathbf{\boldsymbol{\delta}}^{1}\label{eq:backprop1}
	\end{equation}

where :math:`\boldsymbol{\delta}^{l}=\frac{\partial e}{\partial\mathbf{a}^{l}}`.
Moreover the error :math:`\boldsymbol{\delta}^{l}` at layer (or time) :math:`l`
can be expressed relatively to :math:`\boldsymbol{\delta}^{l+1}` for :math:`1\leq l<L`:

.. math::
	\begin{eqnarray}
	\delta_{p}^{l} & = & \frac{\partial e}{\partial a^{l}}\\
	 & = & \underset{i,j}{\sum}\frac{\partial e}{\partial a_{i}^{l+1}}\frac{\partial a_{i}^{l+1}}{\partial x_{j}^{l}}\frac{\partial x_{j}^{l}}{\partial a_{k}^{l}}\\
	 & = & \underset{i}{\sum}\delta_{i}^{l+1}W_{i,p}g'(a_{p}^{l}),
	\end{eqnarray}

or in matrix form:

.. math::
	\begin{equation}
	\boldsymbol{\delta}^{l}=\mathbf{W}{}^{T}\boldsymbol{\delta}^{l+1}\cdot g'(\mathbf{a}^{l})\label{eq:gardvanisheq}
	\end{equation}

Finally :math:`\boldsymbol{\delta}^{L-1}` at the last layer is:

.. math::
	\begin{equation}
	\boldsymbol{\delta}^{L-1}=\frac{\partial e}{\partial\mathbf{a}^{L}}=\frac{\partial\frac{1}{2}(\mathbf{y}^{L-1}-\mathbf{u}^{L})^{T}(\mathbf{y}^{L-1}-\mathbf{u}^{L})}{\partial\mathbf{a}^{L}}=(\mathbf{W}^{out})^{T}\boldsymbol{\delta}^{L}\label{eq:bckprp3}
	\end{equation}
where :math:`\boldsymbol{\delta}^{L}=(\mathbf{y}^{L-1}-\mathbf{u}^{L})`.
Using \ref{eq:backprop1}, \ref{eq:gardvanisheq} and \ref{eq:bckprp3}
we can recursively calculate the update of :math:`\mathbf{f}^{1}`. But
first let us see the form of the error :math:`\boldsymbol{\delta}^{1}`:

.. math::
	\begin{eqnarray}
	\boldsymbol{\delta}^{1} & = & \mathbf{W}{}^{T}\boldsymbol{\delta}^{2}\cdot g'(\mathbf{a}^{2})\\
	 & = & \mathbf{W}{}^{T}\left(\mathbf{W}{}^{T}\boldsymbol{\delta}^{2}\cdot g'(\mathbf{a}^{3})\right)\cdot g'(\mathbf{a}^{2})\\
	 & \vdots\\
	 & = & \mathbf{W}{}^{T}\left(\mathbf{W}{}^{T}\left(\left(\ldots(\mathbf{W}^{out})^{T}\boldsymbol{\delta}^{L}\ldots\right)\cdot g'(\mathbf{a}^{3})\right)\cdot g'(\mathbf{a}^{2})\right)
	\end{eqnarray}
We can use the fact that :math:`g'=\tanh'<1` to find an upper bound on
the Euclidean norm of this error:

.. math::
	\begin{align}
	\Vert\boldsymbol{\delta}^{1}\Vert & \leq\Vert\mathbf{W}\Vert^{L-1}\Vert(\mathbf{W}^{out})\Vert\Vert\boldsymbol{\delta}^{L}\Vert.
	\end{align}
We can rearrange this inequality to make explicit the role of :math:`\bar{\sigma}(\mathbf{W})`,
the largest singular value of :math:`\mathbf{W}`:

.. math::
	\begin{equation}
	\Vert\boldsymbol{\delta}^{1}\Vert\sim\bar{\sigma}(\mathbf{W})^{L}\|\boldsymbol{\delta}^{L}\|,
	\end{equation}
and use the sufficient condition for the echo state property described
in \ref{sub:The-Echo-State}: :math:`\bar{\sigma}(\mathbf{W})<1`. Hence,
for an echo state network :math:`\Vert\boldsymbol{\delta}^{1}\Vert=\bar{\sigma}(\mathbf{W})^{L}\|\boldsymbol{\delta}^{L}\|\to0`
for :math:`L\to+\infty`. 

In summary, after the error has been back-propagated through the recurrent
layers :math:`n` times, the magnitude of the gradient error used to update
the word feature :math:`f(\mathbf{u}_{n})` conditioned on
the prediction of :math:`\mathbf{u}_{t+n}` tends to decrease
geometrically. It is thus possible to approximate the gradient by
taking into account only a limited number of time-steps. In fact we
chose to compute the gradient only taking into account one pass through
the recurrent layer. This allows reducing the computational burden
of the feature update and makes the architecture of the proposed network
close to a simple feed forward network with memory.


Code
----

.. automodule:: esnlm.features.features
   :members:
   :undoc-members:
   
