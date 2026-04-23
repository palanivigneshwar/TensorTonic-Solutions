import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x=np.asarray(x)
    return 1.0/(1.0+np.exp(-x))