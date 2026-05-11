def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    relevant=set(relevant)
    recommended=set(recommended[:k])
    intersect=relevant.intersection(recommended)
    return [len(intersect)/k,len(intersect)/len(relevant)]