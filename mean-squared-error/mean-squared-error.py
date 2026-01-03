import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_true.shape!=y_pred.shape:
        return None
    N = y_true.shape[0]
    sse=np.sum((y_pred-y_true)**2)
    return sse/N
