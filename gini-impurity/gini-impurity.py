import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    print(y_left)
    print(y_right)
    nl=len(y_left)
    nr=len(y_right)
    N=nl+nr
    if N==0:
        return 0.0
    ginil,ginir=1,1
    numbersl,countsl=np.unique(y_left,return_counts=True)
    numbersr,countsr=np.unique(y_right,return_counts=True)

    for _,count in zip(numbersl,countsl):
        ginil-=(count/nl)**2

    for _,count in zip(numbersr,countsr):
        ginir-=(count/nr)**2
        
    GINI=((nl/N)*ginil)+((nr/N)*ginir)
    return GINI