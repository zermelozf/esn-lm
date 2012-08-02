Logistic Regression
===================


First import the LogisticRegression class::

	>>> from esnlm.nodes import LogisticRegression
	
Then construct a dummy model and generate a dummy dataset::

	>>> input_dim, output_dim = 2, 2
	>>> dmodel = LogisticRegression(input_dim, output_dim)
	
	>>> x = np.random.rand(300, input_dim-1)
	>>> x = np.hstack([x, np.ones((x.shape[0], 1))])
	>>> y = dmodel.sample_y_given_x(x)
	
Construct another model and try to learn the first model::

	>>> model = LogisticRegression(input_dim, output_dim)
	>>> model.fit(x,y, method='Newton-Raphson', nb_iter=20)
	
Finally, plot the probability of the different classes::
	
	>>> pyr = dmodel.py_given_x(x)
	>>> py = model.py_given_x(x)
	
	>>> import matplotlib.pyplot as plt
	>>> plt.plot( x[:,0], pyr[:,0], 'x', color='black')
	>>> plt.plot( x[:,0], pyr[:,1], 'x', color='black')
	>>> plt.plot( x[:,0], py[:,0], 'xb')
	>>> plt.plot( x[:,0], py[:,1], 'xg')
	>>> plt.show()
	
.. plot:: pyplots/log_reg.py

   
	
	