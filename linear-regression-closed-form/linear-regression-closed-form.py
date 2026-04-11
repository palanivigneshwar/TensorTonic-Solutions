import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    XT=np.transpose(X)
    cov=XT@X
    XTXI=np.linalg.inv(cov)
    XTy=XT@y
    return XTXI@XTy