import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N,D=X.shape
    w,b=np.zeros(D),0.0
    for _ in range(steps):
        z=np.dot(X,w)+b
        pred=_sigmoid(z)
        error=pred-y
        w-=lr*np.dot(X.T,error)/N
        b-=lr*np.mean(error)
    return [w,b]