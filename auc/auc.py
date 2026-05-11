import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    auc=0
    n=len(tpr)
    for i in range(0,n-1):
        auc+=(tpr[i]+tpr[i+1])*(fpr[i+1]-fpr[i])
    return auc/2