def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    
    centroid=[]
    for i in range(k):
        points_k=[points[j] for j in range(len(assignments)) if assignments[j]==i]
        n=len(points_k)
        if n!=0:
            sum=[0 for j in range(len(points[0]))]
            for point_k in points_k:
                sum=[sum[j]+point_k[j] for j in range(len(points[0]))]
            centroid.append([sums/n for sums in sum])
        else:
            centroid.append([0.0 for i in range(len(points[0]))])
    return centroid