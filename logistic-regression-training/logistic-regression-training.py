import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train a binary logistic regression classifier using gradient descent.
    
    X: np.ndarray of shape (N, D)
    y: np.ndarray of shape (N,)
    Returns: (w, b) where w is shape (D,) and b is a float
    """
    N, D = X.shape
    
    # 1. Initialize weights as zeros (shape D,) and bias as 0.0
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # 2. Forward Pass
        # z = Xw + b (Dot product for N samples)
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 3. Compute error (p - y)
        # Since p and y are both shape (N,), this is element-wise subtraction
        error = p - y
        
        # 4. Compute Gradients (Hint 1)
        # grad_w = (X_transpose . error) / N -> shape (D,)
        grad_w = np.dot(X.T, error) / N
        
        # grad_b = average of the error -> float
        grad_b = np.mean(error)
        
        # 5. Update Parameters (Hint 2)
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, float(b)