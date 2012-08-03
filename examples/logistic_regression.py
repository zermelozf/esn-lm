import numpy as np
from esnlm.readouts import LogisticRegression

# Test Model
idim, odim = 1, 3
dm =LogisticRegression(input_dim=idim+1, output_dim=odim)
x = np.random.rand(50000, idim)
x = np.hstack([x,np.ones((x.shape[0], 1))])
y = dm.sample_y_given_x(x)
print "input_dim =", x.shape[1], ", output_dim =", y.shape[1], "nb_samples:", x.shape[0]


input_dim, output_dim = x.shape[1], y.shape[1]
m = LogisticRegression(input_dim, output_dim)
init_value = m.log_likelihood(x, y)
       
initial_params = np.array(m.params)
m.log_likelihood(x, y)
method = 'Newton-Raphson'
print "... training using", method, ":",
m.fit(x, y, method=method)

print ""
print "Model Ini ->", init_value
print "Model Fin ->", m.log_likelihood(x, y) 
print "Model Rea ->", dm.log_likelihood(x, y)

##############
#### PLOT ####
##############
from matplotlib.pyplot import plot, get_cmap, title, show

py1 = dm.py_given_x(x)
py2 = m.py_given_x(x)

for i in range(py1.shape[1]):
    plot(x[:,0], py1[:, i], 'x', color='black')

cm = get_cmap('gist_rainbow')
NUM_COLORS = py2.shape[1]
for i in range(py2.shape[1]):
    col = cm(1.*i/NUM_COLORS)
    plot(x[:,0], py2[:, i], 'x', color=col)
    
title("Multivariable Logistic Regression")
show()