Newton-Raphson
==============

In the classification setting, both the gate model and the experts
can be estimated minimizing a cross-entropy function. This is a convex
optimization problem that can be solved using the Newton-Raphson method. 

The Netwon-Raphson method approximate the real objective function
:math:`\mathcal{Q}` to minimize or maximize by a quadratic form around
the current point :math:`\mathbf{x}` using its Taylor series expansion:

.. math::
	\begin{equation}
	\mathcal{Q}(\mathbf{x}+\mathbf{\boldsymbol{\epsilon}})\approx\mathcal{Q}(\mathbf{x})+(\nabla\mathcal{Q})^{T}\boldsymbol{\epsilon}+\frac{1}{2}\boldsymbol{\epsilon}^{T}(\nabla^{2}\mathcal{Q})\boldsymbol{\epsilon},
	\end{equation}
This quadratic form is then minimized (or maximized) with respect
to :math:`\boldsymbol{\epsilon}` by finding the zero of its derivative,

.. math::
	\begin{equation}
	\begin{array}{cccc}
	 & \frac{d\mathcal{Q}(\mathbf{x}+\mathbf{\boldsymbol{\epsilon}})}{d\mathbf{\boldsymbol{\epsilon}}} & = & 0\\
	\Leftrightarrow & \nabla\mathcal{Q}+(\nabla^{2}\mathcal{Q})\boldsymbol{\epsilon} & = & 0\\
	\Leftrightarrow & \boldsymbol{\epsilon} & = & -(\nabla^{2}\mathcal{Q})^{-1}(\nabla\mathcal{Q}),
	\end{array}
	\end{equation}
leading to an iterative update of :math:`\mathbf{x}`:

.. math::
	\begin{equation}
	\mathbf{x}_{t+1}=\mathbf{x}_{t}-(\nabla^{2}\mathcal{Q})^{-1}(\nabla\mathcal{Q})
	\end{equation}
If the real objective function is quadratic, this converges in one
step.

Let us now use this method to find the minimum of the cross-entropy
:math:`\mathcal{Q}_{g}` (for the gating model) using its first and second
derivative :math:`\nabla\mathcal{Q}_{g}` and :math:`\nabla^{2}\mathcal{Q}_{g}`. 
Plugging (\ref{eq:gateeq}) into (\ref{eq:gateoptfunc}), the form
of the objective :math:`\mathcal{Q}_{g}` to be optimized in order to find
the parameters of the gating model is:

.. math::
	\begin{eqnarray}
	\mathcal{Q}_{g}(\mathcal{X},\Theta_{g}) & = & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln p(z|\mathbf{x}_{n})\\
	 & = & \sum_{n=1}^{M}\sum_{z=1}^{K}q_{n}(z)\ln\frac{\exp(\xi_{nz})}{\sum_{k=1}^{K}\exp(\xi_{nk})}.
	\end{eqnarray}
Its derivative can be computed by applying the chain rule successively
to the multinomial logit

.. math::
	\begin{eqnarray}
	\frac{\partial\sigma_{i}}{\partial\xi_{j}} & = & \frac{\partial}{\partial\xi_{j}}\left(\frac{\exp(\xi_{i})}{\sum_{k}\exp(\xi_{k})}\right)\\
	 & = & \sigma_{i}(\delta_{i,j}-\sigma_{j})
	\end{eqnarray}
and a linear function

.. math::
	\begin{eqnarray}
	\frac{\partial\xi_{l}}{\partial\Theta_{g,ij}} & = & \frac{\partial}{\partial\Theta_{g,ij}}\left(\sum_{k}x_{k}\Theta_{g,kl}\right)\\
	 & = & \delta_{j,l}x_{i}.
	\end{eqnarray}
This leads to:

.. math::
	\begin{eqnarray}
	\frac{\partial\mathcal{Q}_{g}(\mathcal{X},\mathbf{\Theta}_{g})}{\partial\Theta_{g,ij}} & = & \frac{\partial}{\partial\Theta_{g,ij}}\left(\sum_{n=1}^{M}\sum_{z=1}^{K}q_{nz}ln\ \sigma_{nz}\right)\\
	 & = & \sum_{n=1}^{M}x_{ni}(q_{nj}-\sigma_{nj}),
	\end{eqnarray}
or in vector form:

.. math::
	\begin{equation}
	\nabla_{\mathbf{\Theta}_{g}}\mathcal{Q}_{g}=\sum_{n=1}^{M}(\mathbf{Q}_{n}-\mathbf{Z}_{n})\otimes\mathbf{X}_{n}.
	\end{equation}
The same can be done for the second derivative of the objective function
:math:`\mathcal{Q}_{g}(\mathcal{X},\Theta_{g})`:

.. math::
	\begin{eqnarray}
	\frac{\partial\mathcal{Q}_{g}(\mathcal{X},\mathbf{\Theta}_{g})}{\partial\Theta_{g,ij}\partial\Theta_{g,kl}} & = & \frac{\partial}{\partial\Theta_{g,kl}}\left(\frac{\partial\mathcal{Q}_{g}(\mathcal{X},\mathbf{\Theta}_{g})}{\partial\Theta_{g,ij}}\right)\\
	 & = & \frac{\partial}{\partial\Theta_{g,kl}}\left(\sum_{n=1}^{M}(q_{nj}-\sigma_{nj})x_{ni}\right)\\
	 & = & -\sum_{n=1}^{M}x_{ni}\sigma_{nj}(\delta_{j,l}-\sigma_{nl})x_{nk},
	\end{eqnarray}
or in vector form: 

.. math::
	\begin{equation}
	\nabla_{\mathbf{\Theta}_{g}}^{2}\mathcal{Q}_{g}=-\sum_{n=1}^{M}\mathbf{x}_{n}^{T}\mathbf{x}_{n}\otimes(\mathbf{\Lambda}(\boldsymbol{\sigma}_{n})-\boldsymbol{\sigma}_{n}^{T}\boldsymbol{\sigma}_{n}).
	\end{equation}
The update of the parameter matrix :math:`\mathbf{\Theta}_{g}` can then
be computed by iteratively minimizing a second order Taylor approximation
of the cross-entropy :math:`\mathcal{Q}_{g}` using Newton-Raphson method.
If we assume that the parameter matrix :math:`\boldsymbol{\Theta}_{g}`
is reshaped as a vector and the Hessian is expressed as a matrix,
then:

.. math::
	\begin{equation}
	\mathbf{\Theta}_{g}^{t+1}=\mathbf{\Theta}_{g}^{t}-(\nabla_{\mathbf{\Theta}_{g}}^{2}\mathcal{Q}_{g})^{-1}\nabla_{\mathbf{\Theta}_{g}}\mathcal{Q}_{g}.
	\end{equation}


For the experts, the derivation of the update formula is very similar.
In the regression case, learning the experts amount to solving a weighted
regression problem and can also be done in one step at each iteration.

Code
----

.. automodule:: esnlm.optimization.gradients
   :members:
   :undoc-members:
   

.. automodule:: esnlm.optimization.newton_raphson
   :members:
   :undoc-members:
   
