Supervised Mixture of Experts
=============================


Description
-----------


The mixture of experts model can be simplified by specifying which
expert should predict the next word in a supervised way. Instead of
letting the model cluster the input space according to features it
discovers itself, we can specify the expert the model must rely on
for the prediction of the next word by assigning a value to the :math:`z_{t}`
for every :math:`t`. This actually simplifies the learning properties of
the algorithms: the training is now totally supervised in the sense
that instead of a partially observed dataset :math:`\mathcal{X}=\{\mathbf{X},\mathbf{Y}\}`
the algorithm now can have access to the fully observed dataset :math:`\mathcal{X}=\{\mathbf{X},\mathbf{Y},\mathbf{Z}\}`. 

FIGURE


However, we have to choose a way to specify automatically the expert.
In a sense, this amounts to restricting the kind of feature the algorithm
is paying attention to. For example, we could choose to assign a different
expert according to the position of the current word in the sentence.
In this way, one expert could be a specialist at predicting the beginning
of a sentence while another would be responsible for sentences ending.
Many other linguistic characteristics could be used to choose between
experts; this allows adding hand crafted features for selecting experts.

Another way of selecting the experts is to rely on the word to be
predicted. For example, using the representation developed in the
feature layer, we can cluster words and assign an expert to each cluster
of words. In fact, this trick is used in [Morin2005]_ to speed
up the training of a neural language model: each expert is assigned
to a subset of the vocabulary. The probability of the next word is
computed according to:

.. math::
	\begin{equation}
	P(w_{t+1}|w1\ldots w_{t})=\sum_{w\in V}p(z_{w}|w_{1}\ldots w_{t})p(w|z_{w},w_{1}\ldots w_{t}).
	\end{equation}

Because :math:`p(w|z_{w'})=0` if :math:`w\neq w'` each expert has to focus on
a smaller vocabulary. To minimize the dimensionality of the system
and reduce the training time, each one of the :math:`\sqrt{|V|}` experts
may each focus on :math:`\sqrt{|V|}` words. Thanks to this trick, when
the size of the vocabulary grows like :math:`|V|`, it is possible to train
the gating and experts models on vector of size only :math:`\sqrt{|V|}`.
The gain in training time is very sensible if the vocabulary is large
and this trick allows to train a network efficiently on a huge number
of words in a reasonable time [Mikolov2011a]_, [Mikolov2011]_.

Finally, it is also possible to keep the latent variable in the mixture
of experts model and add another level of nodes to separate the vocabulary.
In fact the architecture can be stacked any number of time as in the
hierarchical mixture of experts model [Jordan1994]_.

Code
----

.. automodule:: esnlm.readouts.smoe
   :members:
   :undoc-members:
   
References
----------

.. [Morin2005] plop
.. [Jordan1994] plop
