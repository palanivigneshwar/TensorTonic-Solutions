import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X=np.asarray(X)
    N=X.shape[0]
    if X.ndim!=2 or N<2:
        return None
    X_mean = np.mean(X,axis=0)
    X_centered = X-X_mean
    X_sum=(X_centered.T@X_centered)/(N-1)
    # print(f"X:{X},X_mean:{X_mean},X_centered:{X_centered},X_sum:{X_sum}")
    return X_sum