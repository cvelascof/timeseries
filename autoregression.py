"""Methods related to autoregressive AR(p) models."""

import numpy as np

def adjust_corrcoefs(gamma_1, gamma_2):
    """A simple adjustment of temporal autocorrelation coefficients to ensure 
    that the resulting AR(2) process is stationary.
    
    Parameters
    ----------
    gamma_1 : float
      Lag-1 temporal autocorrelation coeffient.
    gamma_2 : float
      Lag-2 temporal autocorrelation coeffient.
    
    Returns
    -------
    out : float
      The adjusted lag-2 correlation coefficient.
    """
    gamma_2_old = gamma_2
    gamma_2 = np.max(gamma_2, 2*gamma_1*gamma_2-1)
    gamma_2 = np.max(gamma_2, (3*gamma_1**2-2+2*(1-gamma_1**2)**1.5) / gamma_1**2)
    
    return gamma_2

def estimate_ar_params_yw(GAMMA, include_perturb_term=False):
    """Estimate the parameters of an AR(p) model from the Yule-Walker equations 
    using the given set of autocorrelation coefficients.
    
    Parameters
    ----------
    GAMMA : array_like
      Two-dimensional array of shape (n,p) containing the lag-l, l=1,2,...p, 
      temporal autocorrelation coefficients for n two-dimensional fields. The 
      correlation coefficients are assumed to be in ascending order with respect 
      to time lag.
    include_perturb_term : bool
      If True, calculate the perturbation term coefficients for the AR(p) model.
    
    Returns
    -------
    out : ndarray
      If include_perturb_term is False, an array of shape (n,p) containing the 
      AR(p) parameters for for the lag-p terms for each cascade level. If 
      include_perturb_term is True, the shape of the array is (n,p+1), where the 
      last column contains the perturbation term coefficients.
    """
    n,p = GAMMA.shape
    
    if include_perturb_term:
        PHI = np.empty((n, p+1))
    else:
        PHI = np.empty((n, p))
    
    for k in xrange(n):
        g = np.hstack([[1.0], GAMMA[k, :]])
        G = []
        for j in xrange(p):
            G.append(np.roll(g[:-1], j))
        G = np.array(G)
        phi = np.linalg.solve(G, g[1:].flatten())
        
        # Check that the absolute values of the roots of the characteristic 
        # polynomial are less than one. Otherwise the AR(p) model is not stationary.
        r = np.array([abs(r_) for r_ in roots([1.0 if i == 0 else -phi[i] \
                      for i in xrange(p)])])
        if any(r >= 1):
            raise Exception("nonstationary AR(p) process")
        
        if not include_perturb_term:
            PHI[k, :p] = phi
        else:
            c = 1.0
            for j in xrange(p):
              c -= GAMMA[k, j] * phi[j]
            phi_pert = np.sqrt(c)
            
            # If the expression inside the square root is negative, phi_pert cannot 
            # be computed and it is set to zero instead.
            if not np.isfinite(phi_pert):
                phi_pert = 0.0
            
            PHI[k, :p] = phi
            PHI[k, -1] = phi_pert
    
    return PHI
