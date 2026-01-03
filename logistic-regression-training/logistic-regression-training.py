import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid activation."""
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    """
    Trains a binary logistic regression model via gradient descent.
    
    Args:
        X: Input features of shape (N, D).
        y: Labels of shape (N,) or (N, 1).
        lr: Learning rate for parameter updates.
        steps: Number of gradient descent iterations.
        
    Returns:
        tuple (w, b): Learned weights as (D,) array and bias as a float.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)  # Ensure y is a column vector
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros((D, 1))
    b = 0.0
    
    for _ in range(steps):
        # 1. Forward Pass: Compute linear combination and predictions
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 2. Compute Gradients:
        # dw = (1/N) * X^T * (p - y)
        # db = (1/N) * sum(p - y)
        dw = np.dot(X.T, (p - y)) / N
        db = np.mean(p - y)
        
        # 3. Parameter Updates
        w -= lr * dw
        b -= lr * db
        
    return w.flatten(), float(b)
