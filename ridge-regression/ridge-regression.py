import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    X: shape (N, D), y: shape (N,), lam: float
    """
    # 1. Get dimensions
    X=np.asarray(X)
    N, D = X.shape
    XT = X.T  # Shortcut for transpose
    
    # 2. Compute the stabilized covariance matrix (XTX + lam*I)
    # The identity matrix must be D x D (number of features)
    XTX = XT @ X 
    penalty = lam * np.eye(D)
    
    # 3. Solve for weights
    # We use np.linalg.inv or pinv on the sum
    A_inv = np.linalg.inv(XTX + penalty)
    
    # 4. Final weight calculation: (A_inv) @ XT @ y
    w = A_inv @ XT @ y
    
    return w