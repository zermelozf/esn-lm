import numpy as np
import matplotlib.pyplot as plt
from esnlm.nodes import LogisticRegression

input_dim, output_dim = 2, 2
x = np.random.rand(300, input_dim-1)
x = np.hstack([x, np.ones((x.shape[0], 1))])
dmodel = LogisticRegression(input_dim, output_dim)
pyr = dmodel.py_given_x(x)
y = dmodel.sample_y_given_x(x)
 
model = LogisticRegression(input_dim, output_dim)
model.fit(x,y, method='Newton-Raphson', nb_iter=20)
py = model.py_given_x(x)
plt.plot( x[:,0], pyr[:,0], 'x', color='black')
plt.plot( x[:,0], pyr[:,1], 'x', color='black')
plt.plot( x[:,0], py[:,0], 'xb')
plt.plot( x[:,0], py[:,1], 'xg')
plt.show()