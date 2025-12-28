import numpy as np
def single_leaky_relu(x,alpha):
    return x if x>=0 else alpha*x
def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    if type(x)==int:
        return np.asarray(single_leaky_relu(x,alpha))
    result = [single_leaky_relu(val,alpha) for val in x]
    return np.asarray(result)