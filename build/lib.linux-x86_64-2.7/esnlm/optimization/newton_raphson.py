""" Module for one-step of Newton-Raphson optimization with step halving. """

import numpy as np

def newton_raphson(grad, hessian, initial_params, objective_function):
    """ Performs a one-step Newton-Rapshon optimization.
    
    Parameters
    ----------
    grad : the gradient in vector form
    hessian : the hessian in matrix form
    initial_params : initial condition point
    objective function : the objective to optimize. Must be a function of the parameters in vector form
    
    """
    
    delta = np.linalg.solve(hessian, grad).reshape(initial_params.shape)
    
    stepsize = 1.
    min_stepsize = 1e-2
    initial_objective = objective_function(initial_params)
    while stepsize > min_stepsize:
        T = np.array(initial_params - stepsize*delta)
        if objective_function(T) > initial_objective:
            break
        stepsize = stepsize/2
    if stepsize < min_stepsize:
        print "Newton-Raphson did not work."
        return initial_params
    else:
        return T