import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    if len(x)==1:
        return np.sqrt((x-y)**2)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        return np.sqrt(np.sum((x-y)**2))