import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    return (1-np.exp(-2*np.asarray(x)))/(1+np.exp(-2*np.asarray(x)))