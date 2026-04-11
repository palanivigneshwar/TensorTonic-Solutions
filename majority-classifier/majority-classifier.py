import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    cluster,count=np.unique(y_train,return_counts=True)
    print(cluster[np.argmax(count)])
    return [cluster[np.argmax(count)]]*len(X_test)