"""Methods for computing spatial and temporal correlation of time series of 
two-dimensional fields."""

import numpy as np

def temporal_autocorrelation(X, k, conditional=False, cond_thr=None):
    """Compute lag-l autocorrelation coefficients gamma_l, l=1,2,...k, for a 
    time series of two-dimensional input fields.
    
    Parameters
    ----------
    X : array-like
      Three-dimensional array of shape (n, L, L) containing the time series of 
      n two-dimensional fields of shape (L, L). The input fields are assumed to 
      be in increasing order with respect to time, and the time step is assumed 
      to be regular (i.e. no missing data).
    k : int
      The number of time lags for which to compute the autocorrelation 
      coefficients.
    conditional : bool
      If set to True, compute the correlation coefficients conditionally by 
      excluding the areas where the values are below the given threshold. This 
      requires cond_thr to be set.
    cond_thr : float
      Threshold value for conditional computation of correlation coefficients, 
      see above.
    
    Returns
    -------
    out : ndarray
      Array of length k containing the temporal autocorrelation coefficients 
      for time lags l=1,2,...,k.
    """
    if len(X.shape) != 3:
        raise ValueError("the input X is not three-dimensional array")
    if X.shape[1] != X.shape[2]:
        raise ValueError("the dimensions of the input fields are %dx%d, but square shape expected" % \
                         (X.shape[1], X.shape[2]))
    if conditional and cond_thr is None:
        raise Exception("conditional=True, but cond_thr was not supplied")
    
    GAMMA = np.empty((n, k))
    
    MASK = np.ones((X.shape[1], X.shape[2]), dtype=bool)
    for k in xrange(X.shape[0]):
      MASK = np.logical_and(MASK, np.isfinite(X[k, :, :))
      if conditional:
          MASK = np.logical_and(MASK, X[k, :, :] >= cond_thr)
    
    gamma = []
    for k in xrange(X.shape[0] - 1):
        gamma.append(np.corrcoef(X[-1, :, :][MASK], X[-(k+2), :, :][MASK])[0, 1])
    
    return gamma
