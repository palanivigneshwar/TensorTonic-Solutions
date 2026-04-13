import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x=np.asarray(x)
    return x*(np.exp(x)/(np.exp(x)+1))