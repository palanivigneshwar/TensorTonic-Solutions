import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    _,counts=np.unique(assignments,return_counts=True)
    clusters=[i for i in range(k)]
    left_overs=k-len(counts)
    for i in range(left_overs):
        counts=np.append(counts,0)
    centroids=[]
    print(counts)
    for cluster,count in zip(clusters,counts):
        point=[points[i] for i in range(len(assignments)) if assignments[i]==cluster]
        centroid=list(np.sum(point,axis=0)/count) if count>0 else [0]*len(points[0])
        print(centroid)
        centroids.append(centroid)
    print(centroids)
    return centroids