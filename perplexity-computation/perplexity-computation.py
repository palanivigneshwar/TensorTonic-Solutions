import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    n=len(prob_distributions)
    summation=0
    for i in range(n):
        summation+=np.log(prob_distributions[i][actual_tokens[i]])
    H=-(1.0/n)*summation
    PP=np.exp(H)
    print(PP)
    return PP