"""Methods for computing spatial and temporal correlation of time series of 
two-dimensional fields."""

import numpy as np

def temporal_autocorrelation(X, k):
    """Compute lag-l autocorrelation coefficients gamma_l, l=1,2,...k, for a 
    time series of two-dimensional input fields.
    
    Parameters
    ----------
    X : array-like
      Three-dimensional 
    k : int
      The number of time lags for which to compute the autocorrelation 
      coefficients.
    
    Returns
    -------
    out : ndarray
      
    """
    GAMMA = empty((n, k))
