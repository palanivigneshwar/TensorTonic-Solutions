import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    if x.ndim==1:
        if x.shape[0]==0:
            return []
        max_x=np.max(x)
        exp=np.exp(x-max_x)
        return exp/np.sum(exp)
    # else:
    #     smax=[softmax(X) for X in x]
    #     return smax
    max_x=np.max(x,axis=1,keepdims=True)
    # print(f"x:{x},max_x:{max_x}")
    exp=np.exp(x-max_x)
    exp_sum=np.sum(exp,axis=1,keepdims=True)
    return exp/exp_sum