import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=float)
    if x.ndim>1 or y.ndim>1:
        x=np.flatten(x)
        y=np.flatten(y)
    if (len(x)!=len(y)):
        raise ValueError("Mismatched Lengths")
    return np.dot(x,y)