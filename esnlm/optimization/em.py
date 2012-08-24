import numpy as np
from ..utils import softmax
from .gradients import gradient, hessian
from scipy.optimize import minimize

def expectation_maximization(model, x, y, max_iter=100):
    """ Generalized Expectation-Maximization algorithm. Learns the parameters of the model by 
        finding a local maximum of the lg-likelihood of the data. It can currently 
        be used for learning the parameters of the MixtureOfExperts model.
        
        Parameters
        ----------
        x : array of shape nb_sample*nb_features
        y : array of shape nb_samples*output_dim
        max_iter : the maximum number of iterations
        
        Returns
        -------
        ll : a list containing the log-likelihood at each iteration
        Q1 : a list containing the gate part log-likelihood
        Q2 : a list containing the experts part log-likeklihood
        
        Notes
        -----
        The maximization step uses the Conjugate-Gradient algorithm of scipy.optimize.
    """
    
    print "... Expectation-Maximization:",
    
    # initialize likelihood parts
    lll = []
    Q1l = []
    Q2l = []
    
    for iter in range(max_iter):
        print iter,
        
        # EXPECTATION
        lik_y = model.lik_y_for_every_z(x, y)
        pz = model.pz_given_x(x)
        pz_given_xy = model.pz_given_xy(x, y)
        
        # LIKELIHOOD
        lll.append( np.sum(np.log(np.sum(pz*lik_y, 1))) )
        Q1l.append( np.sum(np.sum(pz_given_xy*np.log(pz), 1)) )
        Q2l.append( np.sum(np.sum(pz_given_xy*np.log(lik_y), 1)) )
        
        # MAXIMIZATION
        # Gates        
        def gates_objective(params):
            lik = np.prod( softmax(np.dot(x, params))**pz_given_xy, axis=1)
            return np.sum(np.log( lik+1e-7  ))           
        
        def obj(params):
            return -gates_objective(params.reshape(model.gates.params.shape))
                
        def grd(params):
            post = softmax(np.dot(x, params.reshape(model.gates.params.shape)))
            return -gradient(x, pz_given_xy, post, np.ones((pz_given_xy.shape[0], ))).squeeze()
        
        def hsn(params):
            post = softmax(np.dot(x, params.reshape(model.gates.params.shape)))
            return -hessian(x, pz_given_xy, post, np.ones((pz_given_xy.shape[0], )))
        
        params = model.gates.params.reshape(model.gates.params.size)
        params = minimize(obj, params, jac=grd, method='CG', options={'maxiter':5}).x
        model.gates.params = params.reshape(model.gates.params.shape)
        
        # Experts
        for z in range(model.nb_experts):
            w = pz_given_xy[:, z]
            
            def experts_objective(params):
                lik = np.sum( y*np.log(softmax(np.dot(x, params))), axis=1)
                return np.sum(pz_given_xy[:, z]*lik)
            
            def obj(params):
                return -experts_objective(params.reshape(model.experts[z].params.shape))
                
            def grd(params):
                model.experts[z].params = np.array(params.reshape(model.experts[z].params.shape))
                post = model.py_given_xz(x, z)
                return -gradient(x, y, post, w).squeeze()
            
            def hsn(params):
                model.experts[z].params = np.array(params.reshape(model.experts[z].params.shape))
                post = model.py_given_xz(x, z)                
                return -hessian(x, y, post, w)

            params = model.experts[z].params.reshape(model.experts[z].params.size)
            params = minimize(obj, params, jac=grd, method='CG', options={'maxiter':5}).x
            model.experts[z].params = params.reshape(model.experts[z].params.shape)
            
        # VERIFICATION
        lik_y = model.lik_y_for_every_z(x, y)
        pz = model.pz_given_x(x)
        
        ll = np.sum(np.log(np.sum(pz*lik_y, 1)))
        Q1 = np.sum(np.sum(pz_given_xy*np.log(pz), 1))
        Q2 = np.sum(np.sum(pz_given_xy*np.log(lik_y), 1))
        
        if ll < lll[iter]:
            print "Big problem at iter", iter, "Previous ll:", lll[iter], "Actual ll:", ll
            
        if Q1 < Q1l[iter]:
            print "Bad NR (Q1) at", iter, "Previous:", Q1l[iter], "Actual:", Q1
            
        if Q2 < Q2l[iter]:
            print "Bad NR (Q2) at iter", iter, "Previous:", Q2l[iter], "Actual:", Q2
            
        # Stop Criterion
        if abs(ll - lll[iter]) < 1:
            break
        
    print "The End." 
    return lll, Q1l, Q2l


from ..optimization import newton_raphson
def expectation_maximization2(model, x, y, max_iter=100):
    """ The Generalized Expectation-Maximization algorithm without relying on scipy.optimize.
        The Newton-Raphson method is used during the maximization step. This can be slow for 
        data that is high dimensional."""
    
    # initialize likelihood parts
    lll = []
    Q1l = []
    Q2l = []
    
    print "...... EM with Newton-Raphson",
    for iter in range(max_iter):
        print iter,
        
        # EXPECTATION
        lik_y = model.lik_y_for_every_z(x, y)
        pz = model.pz_given_x(x)
        pz_given_xy = model.pz_given_xy(x, y)
        
        # LIKELIHOOD
        lll.append( np.sum(np.log(np.sum(pz*lik_y, 1))) )
        Q1l.append( np.sum(np.sum(pz_given_xy*np.log(pz), 1)) )
        Q2l.append( np.sum(np.sum(pz_given_xy*np.log(lik_y), 1)) )
        
        # MAXIMIZATION
        # Gates
        w = np.ones((y.shape[0], ))
        
        def gates_objective(params):
            lik = np.prod( softmax(np.dot(x, params))**pz_given_xy, axis=1)
            return np.sum(np.log( lik+1e-7  ))           
        
        grad = gradient(x, pz_given_xy, pz, w)
        hess = hessian(x, pz_given_xy, pz, w)
        
        params = np.array(model.gates.params)
        model.gates.params = newton_raphson(grad, hess, params, gates_objective)
        
        # Experts
        for z in range(model.nb_experts):
            py_given_xz = model.py_given_xz(x, z)
            w = pz_given_xy[:, z]
            
            def experts_objective(params):
                lik = np.sum( y*np.log(softmax(np.dot(x, params))), axis=1)
                return np.sum(pz_given_xy[:, z]*lik)
            
            grad = gradient(x, y, py_given_xz, w)
            hess = hessian(x, y, py_given_xz, w)

            params = np.array(model.experts[z].params)
            model.experts[z].params = newton_raphson(grad, hess, params, experts_objective)
            
        # VERIFICATION
        lik_y = model.lik_y_for_every_z(x, y)
        pz = model.pz_given_x(x)
        
        ll = np.sum(np.log(np.sum(pz*lik_y, 1)))
        Q1 = np.sum(np.sum(pz_given_xy*np.log(pz), 1))
        Q2 = np.sum(np.sum(pz_given_xy*np.log(lik_y), 1))
        
        if ll < lll[iter]:
            print "Big problem at iter", iter, "Previous ll:", lll[iter], "Actual ll:", ll
            
        if Q1 < Q1l[iter]:
            print "Bad NR (Q1) at", iter, "Previous:", Q1l[iter], "Actual:", Q1
            
        if Q2 < Q2l[iter]:
            print "Bad NR (Q2) at iter", iter, "Previous:", Q2l[iter], "Actual:", Q2
            
        # Stop Criterion
        if abs(ll - lll[iter]) < 1:
            break
        
    print "The End." 
    return lll, Q1l, Q2l

