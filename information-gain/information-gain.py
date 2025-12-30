import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    N=len(y)
    if N == 0 or split_mask.size==0:
        return 0.0
    hy=_entropy(split_mask)
    split_mask=np.asarray(split_mask)
    IG=hy
    unique_classes,counts=np.unique(y,return_counts=True)
    for (unique_class,count) in zip(unique_classes,counts):
        unique_class_labels=split_mask[y==unique_class]
        IG-=float(count/N)*_entropy(unique_class_labels)
    return IG

