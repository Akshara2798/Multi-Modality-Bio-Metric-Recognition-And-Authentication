import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import os

def initialize_clusters(frames):
    # Initialize each frame as a 1D vector
    clusters = []
    for frame in frames:
        clusters.append(np.reshape(frame, -1))
        print(np.reshape(frame, -1))
        print(frame)
    # clusters = [np.reshape(frame, -1) for frame in frames]
    return clusters

def calculate_optimal_k(clusters):
    # Initialize an empty list to store the distortions for each k
    distortions = []
    
    # Get the total number of samples in the clusters
    n_samples = len(clusters)
    
    # Set the maximum number of clusters to the smaller of 20 or the number of samples
    max_k = min(20, n_samples)  # Limit the number of clusters to 20 or less if there are fewer samples
    
    # Generate a range of k values to test (from 2 to max_k inclusive)
    K = range(2, max_k + 1)  # Test for k from 2 to max_k
    
    # Loop over each value of k in the range
    for k in K:
        # Apply the KMeans algorithm with k clusters, fitting it to the data (clusters)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(clusters)
        
        # Append the inertia (sum of squared distances to the closest cluster center) to the distortions list
        distortions.append(kmeans.inertia_)
    
    # Find the value of k that minimizes the distortion (inertia) to determine the optimal number of clusters
    optimal_k = K[np.argmin(distortions)]
    
    # Return the optimal number of clusters
    return optimal_k



def merge_clusters(clusters, k):
    while len(clusters) > k:
        # Calculate pairwise distances between clusters
        distances = cdist(clusters, clusters)
        np.fill_diagonal(distances, np.inf)  # Exclude self-distances
        
        # Find the indices of the closest clusters
        min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
        merge_idx1, merge_idx2 = min_dist_idx
        
        # Merge the two closest clusters
        merged_cluster = (clusters[merge_idx1] + clusters[merge_idx2]) / 2
        clusters[merge_idx1] = merged_cluster
        
        # Remove the second cluster from the list
        clusters.pop(merge_idx2)
    
    return clusters

def select_key_frames(clusters, frames):
    key_frames = []
    for cluster in clusters:
        # Find the frame closest to the cluster center
        distances = cdist([cluster], [np.reshape(frame, -1) for frame in frames])
        closest_frame_idx = np.argmin(distances)
        key_frames.append(frames[closest_frame_idx])
    return key_frames



# def select_key_frames(clusters, frames, num_frames_per_cluster=10):
#     key_frames = []
#     for cluster in clusters:
#         # Calculate distances of all frames to the cluster center
#         distances = cdist([cluster], [np.reshape(frame, -1) for frame in frames])
#         # Sort distances and select 'num_frames_per_cluster' closest frames
#         closest_frame_idxs = np.argsort(distances[0])[:num_frames_per_cluster]
#         for idx in closest_frame_idxs:
#             key_frames.append(frames[idx])
#     return key_frames


