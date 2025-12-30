import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X=np.asarray(X)
    if X.ndim==1:
        # print(X)
        X=X.reshape(-1,1)
        # print(X)
        # print(X.max(axis=0,keepdims=True))
    X_max=X.max(axis,keepdims=True)
    X_min=X.min(axis,keepdims=True)
    # print(f"X:{X},X_max:{X_max},X_min:{X_min}")
    X_new=(X-X_min)/np.maximum((X_max-X_min),eps)
    # print(f"X:{X},X_max:{X_max},X_min:{X_min},X_new:{X_new}")
    # print(X_new)
    return X_new