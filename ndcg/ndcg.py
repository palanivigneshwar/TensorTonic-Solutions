import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    sorted_relevance_scores=sorted(relevance_scores)[::-1]
    k=min(k,len(relevance_scores))
    DCG=0
    IDCG=0
    for idx in range(1,k+1):
        DCG+=(math.pow(2,relevance_scores[idx-1])-1)/(math.log2(idx+1)) if (math.log2(idx+1)) else 1.0
        IDCG+=(math.pow(2,sorted_relevance_scores[idx-1])-1)/(math.log2(idx+1)) if (math.log2(idx+1)) else 1.0
    return DCG/IDCG if IDCG!=0 else 0.0