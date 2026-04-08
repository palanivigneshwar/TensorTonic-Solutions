import numpy as np

def knn_distance(X_train, X_test, k):
    # 1. Standardize inputs to 2D arrays (N, D)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
        
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # 2. Vectorized Distance Computation (Hint 1)
    # This creates a (n_test, n_train, d) tensor of differences
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    
    # Calculate Euclidean distance: Square -> Sum -> Sqrt
    # Summing over axis 2 (the features)
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # 3. Sort indices by distance (Hint 2)
    sorted_indices = np.argsort(distances, axis=1)

    # 4. Handle k larger than n_train (Requirement)
    if k > n_train:
        # Create a result array filled with -1
        res = np.full((n_test, k), -1, dtype=int)
        # Fill the first n_train columns with the actual sorted indices
        res[:, :n_train] = sorted_indices
    else:
        # Just take the first k neighbors
        res = sorted_indices[:, :k]

    return res