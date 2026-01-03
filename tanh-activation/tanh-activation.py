import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    if type(x)==int or type(x)==float:
        x=[x]
    x = np.asarray(x)
    exp = np.exp(-2*x)
    numerator = 1-exp
    denominator = 1+exp
    return np.asarray(numerator/denominator,dtype=float)