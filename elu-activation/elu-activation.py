import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    return list(np.where(np.asarray(x)>0,x,alpha*(np.exp(x)-1)))