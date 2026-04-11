def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    ans=[]
    for point in points:
        distance=[]
        for centroid in centroids:
            distance.append(sum([(p-c)**2 for p,c in zip(point,centroid)]))
        ans.append(distance.index(min(distance)))
    return ans