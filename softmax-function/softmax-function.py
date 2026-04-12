import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x=np.asarray(x)
    try:
        print(x.shape)
        N,D=x.shape
        print(f"{N}X{D}")
        ans=[]
        for X in x:
            ans.append(np.exp(X-np.max(X))/np.sum(np.exp(X-np.max(X))))
        return ans
    except:
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
        