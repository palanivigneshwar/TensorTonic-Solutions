import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x=np.asarray(x)
    np_erf=np.vectorize(math.erf)
    return (0.5)*x*(1+np_erf(x/(2**(0.5))))
