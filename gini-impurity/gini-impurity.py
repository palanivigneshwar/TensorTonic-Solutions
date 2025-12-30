import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    len_left=len(y_left)
    len_right=len(y_right)
    N=len_left+len_right
    if N == 0:
        return 0.0
    if len_left==0:
        gini_left=0.0
    else:
        _,count_left=np.unique(y_left,return_counts=True)
        p_left=0
        for count in count_left:
            p_left+=float(count/len_left)**2
        gini_left=1.0-p_left
    if len_right==0:
        gini_right=0.0
    else:
        _,count_right=np.unique(y_right,return_counts=True)
        p_right=0
        for count in count_right:
            p_right+=float(count/len_right)**2
        gini_right=1.0-p_right
    gini_split=((len_left/N)*gini_left)+((len_right/N)*gini_right)
    return float(gini_split)
