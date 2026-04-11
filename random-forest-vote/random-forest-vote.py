import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code 
    results=[]
    for prediction in np.array(predictions).T:
        lables,counts=np.unique(prediction,return_counts=True)
        results.append(lables[np.argmax(counts)])
    return results