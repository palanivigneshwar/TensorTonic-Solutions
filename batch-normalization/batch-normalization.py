import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    # print(f"x:{x},mean:{mean},var:{var}")
    # print(f"x-mean:{(x-mean)},np.sqrt:{np.sqrt(var-eps)}")
    # print(f"new_x:{new_x}")
    if x.ndim==4:
        axes = (0, 2, 3)
        mean=np.mean(x,axis=axes,keepdims=True)
        var=np.var(x,axis=axes,keepdims=True)
        new_x=np.divide((x-mean),np.sqrt(var+eps))
        y=gamma.reshape((1,-1,1,1))*new_x+beta.reshape((1,-1,1,1))
    else:
        mean=np.mean(x,axis=0,keepdims=True)
        var=np.var(x,axis=0,keepdims=True)
        new_x=np.divide((x-mean),np.sqrt(var+eps))
        y=gamma*new_x+beta
    return y