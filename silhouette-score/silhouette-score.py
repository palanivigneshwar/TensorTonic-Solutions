import numpy as np

def euclidean_distance(X,Y):
    return np.sqrt(np.sum((X-Y)**2))

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X=np.asarray(X)
    labels=np.asarray(labels)
    unique_labels,counts = np.unique(labels,return_counts=True)
    s=[]
    for idx,x in enumerate(X):
        same_label_data = X[labels==labels[idx]]
        same_label_distance = [euclidean_distance(x,y) for y in same_label_data if euclidean_distance(x,y)!=0]
        a = np.mean(same_label_distance)
        different_labels = [label for label in unique_labels if label!=labels[idx]]
        b=float('inf')
        for label in different_labels:
            different_label_data = X[labels==label]
            different_label_distance = [euclidean_distance(x,y) for y in different_label_data]
            b=min(b,np.mean(different_label_distance))
        s.append((b-a)/max(b,a))
    return np.mean(s)