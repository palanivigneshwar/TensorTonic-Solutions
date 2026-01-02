import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_mean = np.mean(y_true)
    ssem = np.sum((y_true-y_mean)**2)
    ssep = np.sum((y_true-y_pred)**2)
    if ssem==0:
      return 1.0 if ssep==0.0 else 0.0
    r2 = 1.0-ssep/ssem
    return r2