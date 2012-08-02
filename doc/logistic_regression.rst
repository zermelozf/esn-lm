Logistic Regression
===================


First import the LogisticRegression class::

	>>> from esnlm.nodes import LogisticRegression
	
Then construct a dummy model and generate a dummy dataset::

	>>> input_dim, output_dim = 5, 5
	>>> dmodel = LogisticRegression(input_dim, output_dim)
	
	>>> x = np.random.rand(1000, input_dim-1)
	>>> x = np.hstack([x, np.ones((x.shape[0], 1))])
	>>> y = dmodel.sample_y_given_x(x)
	
Construct another model and try to learn the first model::

	>>> model = LogisticRegression(input_dim, output_dim)
	>>> model.fit(x,y, method='Newton-Raphson', nb_iter=20)
	
	