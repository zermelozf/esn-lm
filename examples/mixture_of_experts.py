import numpy as np
from esnlm.readouts import MixtureOfExperts

idim, nbexp, odim = 10, 3, 10
dmoe = MixtureOfExperts(idim+1, nbexp, odim)
x = np.random.rand(10000, idim)
x = np.hstack([x,np.ones((x.shape[0], 1))])
y = dmoe.sample_y_given_x(x)
print "input_dim =", x.shape[1], ", output_dim =", y.shape[1], ", nb_experts =", nbexp, ", nb_samples:", x.shape[0]

print "... training" 
input_dim, nb_experts, output_dim = x.shape[1], nbexp, y.shape[1]
moe = MixtureOfExperts(input_dim, nb_experts, output_dim)
ll, Q1, Q2 = moe.fit(x, y, max_iter=10)

print "Model fin ->", ll[-1]
print "Model rea ->", dmoe.log_likelihood(x, y)

from matplotlib.pyplot import plot, figure, legend, title, get_cmap, colors, show

figure(1)
p1, = plot(range(len(ll)), ll, 'r')
p2, = plot(range(len(Q1)), Q1, 'b')
p3, = plot(range(len(Q2)), Q2, 'g')
title("Likelihood")
legend([p1, p2, p3], ["log-likelihood", "gates-likelihood", "experts-likelihood"], loc="lower right")
show()