import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Handle the case where p is 1.0 (though constraints say p < 1.0)
    if p >= 1.0:
        return np.zeros_like(x), np.zeros_like(x)
        
    # Determine the keep probability
    keep_prob = 1.0 - p
    
    # Generate random values using the provided rng or np.random
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)
        
    # Create a binary mask where 1 indicates "keep"
    # We use < keep_prob to ensure p is the probability of dropping
    mask = (random_vals < keep_prob)
    
    # Create the dropout pattern with scaling: 1 / (1 - p) for kept elements
    scale = 1.0 / keep_prob
    dropout_pattern = mask.astype(float) * scale
    
    # Apply the pattern to the input
    output = x * dropout_pattern
    
    return output, dropout_pattern
