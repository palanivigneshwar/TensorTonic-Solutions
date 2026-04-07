import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if y==[]:
        return 0.0
    pos_sum=np.sum(y)
    numbers,counts=np.unique(y,return_counts=True)
    n=len(y)
    entropy=0
    for _,count in zip(numbers,counts):
        pi=count/n
        pilogpi=pi*np.log2(pi)
        entropy-=pilogpi
    return entropy