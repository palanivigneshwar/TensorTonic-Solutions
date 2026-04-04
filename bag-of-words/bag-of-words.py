import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    ans=np.zeros(len(vocab),dtype=int)
    dict_vocab={}
    dict_vocab={word:idx for idx,word in enumerate(vocab)}
    for idx,token in enumerate(tokens):
        if token in dict_vocab:
            index=dict_vocab[token]
            ans[index]+=1
    return ans