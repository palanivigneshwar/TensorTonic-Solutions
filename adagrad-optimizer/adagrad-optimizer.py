import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    G=G+np.pow(g,2)
    w=w-(lr/np.sqrt(G+eps))*g
    return (w,G)