import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams

def compute_persistence(X, maxdim=2):
    diagrams = []
    for window in X:
        # Treat each window as a point cloud in PCA space
        result = ripser(window.reshape(-1, 1), maxdim=maxdim)
        dgms = result['dgms']
        
        # Replace infinite death times with the max value in the window
        max_val = window.max()
        for dim_dgm in dgms:
            # The second column [:, 1] is the death time
            dim_dgm[np.isinf(dim_dgm[:, 1]), 1] = max_val
            
        diagrams.append(dgms)
    return diagrams


def vectorize_diagrams(diagrams):
    features = []
    for dgms in diagrams:
        window_features = []
        for dim_dgm in dgms:
            # Filter out points with zero lifetime (birth == death)
            persistent_features = dim_dgm[dim_dgm[:, 1] > dim_dgm[:, 0]]
            
            if len(persistent_features) == 0:  # Handle cases with no persistent features
                window_features.extend([0, 0, 0])
                continue
            
            lifetimes = persistent_features[:, 1] - persistent_features[:, 0]
            window_features.extend([lifetimes.sum(), lifetimes.mean(), lifetimes.max()])
        features.append(window_features)
    return np.array(features)


def plot_example_barcode(diagram):
    plot_diagrams(diagram, plot_bars=True)
