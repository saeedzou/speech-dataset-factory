import numpy as np
import os
import argparse
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from typing import List

def cluster_embeddings(emb_dir: str, eps: float, min_samples: int, embedding_dim: int) -> None:
    """
    Loads speaker embeddings from a directory, clusters them using DBSCAN with a 
    cosine metric, and reports the estimated number of unique speakers.

    Args:
        emb_dir (str): Directory containing speaker embedding .npy files.
        eps (float): The maximum distance (epsilon) for DBSCAN. This is a critical
                     hyperparameter for controlling cluster density.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.
        embedding_dim (int): The expected dimension of the embeddings.
    """
    # 1. Collect all valid embeddings from the specified directory
    all_embeddings: List[np.ndarray] = []
    print(f"Scanning directory: {emb_dir}")
    for fname in sorted(os.listdir(emb_dir)):
        if fname.endswith(".npy"):
            fpath = os.path.join(emb_dir, fname)
            try:
                arr = np.load(fpath)
                # Validate shape and append
                if arr.ndim == 2 and arr.shape[1] == embedding_dim:
                    all_embeddings.append(arr)
                else:
                    print(f"  - Skipping {fname}: unexpected shape {arr.shape}. Expected (N, {embedding_dim}).")
            except Exception as e:
                print(f"  - Could not load or process {fname}: {e}")

    if not all_embeddings:
        print("No valid embedding files found. Exiting.")
        return

    # 2. Vertically stack all loaded embeddings into a single matrix
    combined_embeddings = np.vstack(all_embeddings)
    print(f"\nLoaded a total of {combined_embeddings.shape[0]} embeddings.")

    # 3. L2-normalize embeddings. For unit vectors, Euclidean distance is monotonically
    #    related to cosine similarity, but we'll stick to the 'cosine' metric for clarity.
    normalized_embeddings = normalize(combined_embeddings)

    # 4. Perform DBSCAN clustering
    print(f"Running DBSCAN with eps={eps} and min_samples={min_samples}...")
    # Using n_jobs=-1 to utilize all available CPU cores
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1).fit(normalized_embeddings)

    labels = clustering.labels_
    
    # 5. Calculate and report the number of unique clusters (speakers)
    # The label -1 is assigned by DBSCAN to outliers/noise points.
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)

    print("\n--- Clustering Results ---")
    print(f"Estimated unique speakers (clusters): {num_clusters}")
    print(f"Outliers/Noise points detected: {num_noise}")
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster speaker embeddings using DBSCAN to estimate the number of unique speakers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--emb_dir',
        type=str,
        required=True,
        help='Directory containing speaker embedding .npy files.'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='The epsilon value for DBSCAN, defining the neighborhood radius. Tune this based on embedding space density.'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=2,
        help='The minimum number of samples for a point to be considered a core point.'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=256,
        help='The expected feature dimension of the embeddings.'
    )

    args = parser.parse_args()
    
    cluster_embeddings(args.emb_dir, args.eps, args.min_samples, args.embedding_dim)