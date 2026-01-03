import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    if type(x)==int or type(x)==float:
        x=[x]
    x=np.asarray(x,dtype=float)
    return np.where(x>=0.0,x,0.0)