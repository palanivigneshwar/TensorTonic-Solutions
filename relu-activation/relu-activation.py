import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    return np.where(np.asarray(x)>0,x,0)