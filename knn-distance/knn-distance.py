import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise Euclidean distances and return k nearest neighbor indices.
    """
    # Convert inputs to numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    
    # Requirement: Support 1D and multi-dimensional features
    # Hint 3: Handle 1D arrays by reshaping to 2D (n_samples, 1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Hint 1: Use broadcasting to compute all pairwise differences
    # Shape becomes (n_test, n_train, n_features)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]

    # Euclidean Distance: sqrt(sum of squared differences across features)
    # Sum along axis 2 (the feature dimension)
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # Hint 2: Use np.argsort() to get indices of sorted distances
    # Sort along axis 1 (the distances to each training point for every test point)
    sorted_indices = np.argsort(dist, axis=1)

    # Extract the first k neighbors
    # Requirement: Sort neighbors by distance (closest first)
    # If k > n_train, take all available first, then pad
    actual_k = min(k, n_train)
    neighbor_indices = sorted_indices[:, :actual_k]

    # Requirement: Handle k larger than training set size (pad with -1)
    if k > n_train:
        padding = np.full((n_test, k - n_train), -1, dtype=int)
        neighbor_indices = np.hstack([neighbor_indices, padding])

    # Requirement: Return numpy array of shape (n_test, k) with integer dtype
    return neighbor_indices.astype(int)
