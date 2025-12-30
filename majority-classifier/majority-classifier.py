import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train=np.asarray(y_train)
    X_test=np.asarray(X_test)
    if y_train.size == 0:
        return np.array([], dtype=int)
    if y_train.size == 1:
        majority_class = y_train[0]
    unique,first_indexes,count=np.unique(y_train, return_index=True, return_counts=True)
    max_count=np.max(count)
    tied_max_value=np.where(count==max_count)[0]
    if len(tied_max_value)==1:
        highest_count=unique[tied_max_value]
    else:
        highest_count=unique[tied_max_value[np.argmin(first_indexes[tied_max_value])]]
    return np.full(X_test.shape[0],highest_count, dtype=int)