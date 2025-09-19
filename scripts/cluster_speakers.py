import numpy as np
import argparse
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Tuple
from collections import Counter

def cluster_and_update_manifest(
    manifest_path: str,
    output_manifest_path: str,
    eps: float,
    min_samples: int,
    embedding_dim: int,
    weight_by_utterances: bool = True
) -> None:
    """
    Loads speaker embeddings based on a JSON manifest, clusters them using DBSCAN,
    assigns a unique speaker ID to each individual speaker segment based on its cluster,
    and writes an updated manifest. Optionally weights embeddings by utterance count.

    Args:
        manifest_path (str): Path to the input JSON manifest file.
        output_manifest_path (str): Path to write the output JSON manifest with the
                                    added 'unique_speaker_id' field.
        eps (float): The maximum distance (epsilon) for DBSCAN. This is a critical
                     hyperparameter for controlling cluster density.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.
        embedding_dim (int): The expected dimension of the embeddings.
        weight_by_utterances (bool): If True, weight embeddings by their utterance count.
    """
    # 1. Read the manifest and count utterances per (embedding_file, speaker_id)
    print(f"Reading manifest from: {manifest_path}")
    manifest_data: List[Dict[str, Any]] = []
    unique_embedding_files = set()
    utterance_counts: Dict[Tuple[str, int], int] = Counter()
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                manifest_data.append(entry)
                if 'embedding_file' in entry and 'speaker_id' in entry:
                    embedding_file = entry['embedding_file']
                    speaker_id = int(entry['speaker_id'])
                    unique_embedding_files.add(embedding_file)
                    # Count utterances for each (embedding_file, speaker_id) pair
                    utterance_counts[(embedding_file, speaker_id)] += 1
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {manifest_path}")
        return

    if not unique_embedding_files:
        print("No 'embedding_file' entries found in the manifest. Exiting.")
        return

    print(f"Found {len(utterance_counts)} unique (embedding_file, speaker_id) pairs")
    if weight_by_utterances:
        print(f"Utterance count statistics: min={min(utterance_counts.values())}, "
              f"max={max(utterance_counts.values())}, "
              f"mean={np.mean(list(utterance_counts.values())):.2f}")

    # 2. Load all unique embeddings and create weighted samples
    all_embeddings_list: List[np.ndarray] = []
    embedding_source_map: List[Tuple[str, int]] = []
    sample_weights: List[float] = []
    
    print(f"Loading {len(unique_embedding_files)} unique embedding files...")
    for fpath in sorted(list(unique_embedding_files)):
        try:
            arr = np.load(fpath)
            if arr.ndim == 2 and arr.shape[1] == embedding_dim:
                all_embeddings_list.append(arr)
                # Create mapping and weights for each vector in the file
                for i in range(arr.shape[0]):
                    embedding_source_map.append((fpath, i))
                    if weight_by_utterances:
                        # Weight by number of utterances for this embedding
                        weight = utterance_counts.get((fpath, i), 1)
                        sample_weights.append(float(weight))
                    else:
                        sample_weights.append(1.0)
            else:
                print(f"  - Skipping {fpath}: unexpected shape {arr.shape}. Expected (N, {embedding_dim}).")
        except Exception as e:
            print(f"  - Could not load or process {fpath}: {e}")

    if not all_embeddings_list:
        print("No valid embeddings were loaded. Exiting.")
        return

    # 3. Stack embeddings and apply weighting strategy
    combined_embeddings = np.vstack(all_embeddings_list)
    sample_weights = np.array(sample_weights)
    
    print(f"\nLoaded a total of {combined_embeddings.shape[0]} embeddings for clustering.")
    
    if weight_by_utterances:
        print("Applying utterance-based weighting strategy...")
        # Approach 1: Duplicate embeddings based on utterance count
        # This effectively gives more "votes" to embeddings with more utterances
        weighted_embeddings = []
        weighted_source_map = []
        
        for i, (embedding, weight) in enumerate(zip(combined_embeddings, sample_weights)):
            # Duplicate the embedding proportional to its weight
            # Use square root to prevent excessive weighting of very frequent speakers
            num_duplicates = max(1, int(np.sqrt(weight)))
            for _ in range(num_duplicates):
                weighted_embeddings.append(embedding)
                weighted_source_map.append(embedding_source_map[i])
        
        combined_embeddings = np.array(weighted_embeddings)
        embedding_source_map = weighted_source_map
        
        print(f"After weighting: {combined_embeddings.shape[0]} embeddings "
              f"(expansion factor: {len(weighted_embeddings) / len(sample_weights):.2f}x)")

    # 4. Normalize and cluster the embeddings
    normalized_embeddings = normalize(combined_embeddings)

    print(f"Running DBSCAN with eps={eps} and min_samples={min_samples}...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1).fit(normalized_embeddings)
    labels = clustering.labels_

    # 5. Map back to original embeddings and assign unique IDs for noise
    if weight_by_utterances:
        # Create a mapping from original embedding index to cluster labels
        original_to_clusters: Dict[Tuple[str, int], List[int]] = {}
        for source, label in zip(embedding_source_map, labels):
            if source not in original_to_clusters:
                original_to_clusters[source] = []
            original_to_clusters[source].append(label)
        
        # For each original embedding, take the most common cluster label
        embedding_to_cluster_id: Dict[Tuple[str, int], int] = {}
        for source, cluster_labels in original_to_clusters.items():
            # Take majority vote, with -1 (noise) having lower priority
            label_counts = Counter(cluster_labels)
            # If there's a tie and one option is -1, prefer the non-noise label
            most_common_label = label_counts.most_common(1)[0][0]
            if len(label_counts) > 1 and most_common_label == -1:
                # Check if there's a non-noise alternative
                non_noise_labels = [l for l in cluster_labels if l != -1]
                if non_noise_labels:
                    most_common_label = Counter(non_noise_labels).most_common(1)[0][0]
            
            embedding_to_cluster_id[source] = int(most_common_label)
    else:
        # Direct mapping without weighting
        embedding_to_cluster_id: Dict[Tuple[str, int], int] = {
            source: int(label)
            for source, label in zip(embedding_source_map, labels)
        }

    # 6. Assign unique IDs for noise/outliers
    # First, find the maximum cluster ID to start assigning new unique IDs
    valid_cluster_ids = [cid for cid in embedding_to_cluster_id.values() if cid != -1]
    next_unique_id = max(valid_cluster_ids) + 1 if valid_cluster_ids else 0
    
    # Create unique IDs for each noise point based on (embedding_file, speaker_id)
    noise_to_unique_id: Dict[Tuple[str, int], int] = {}
    
    for source, cluster_id in embedding_to_cluster_id.items():
        if cluster_id == -1:  # This is a noise/outlier point
            if source not in noise_to_unique_id:
                # Assign a new unique ID for this specific (embedding_file, speaker_id)
                noise_to_unique_id[source] = next_unique_id
                next_unique_id += 1
            # Update the mapping with the new unique ID
            embedding_to_cluster_id[source] = noise_to_unique_id[source]

    # Report clustering statistics
    final_labels = list(embedding_to_cluster_id.values())
    # Count actual clusters (non-negative IDs that came from DBSCAN clustering)
    dbscan_clusters = set(label for label in final_labels if label >= 0 and label in set(labels) and label != -1)
    num_dbscan_clusters = len(dbscan_clusters)
    
    # Count unique noise assignments
    original_noise_count = sum(1 for label in labels if label == -1)
    unique_noise_assignments = len(noise_to_unique_id)
    
    total_unique_speakers = len(set(final_labels))
    
    print("\n--- Clustering Results ---")
    print(f"DBSCAN clusters found: {num_dbscan_clusters}")
    print(f"Original noise/outlier points: {original_noise_count}")
    print(f"Unique noise assignments created: {unique_noise_assignments}")
    print(f"Total estimated unique speakers: {total_unique_speakers}")
    print(f"Weighting strategy: {'Utterance-based' if weight_by_utterances else 'Uniform'}")
    print("--------------------------\n")

    # 7. Write the new manifest with the 'unique_speaker_id' field
    print(f"Writing updated manifest to: {output_manifest_path}")
    with open(output_manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_data:
            embedding_file = entry.get('embedding_file')
            speaker_id = entry.get('speaker_id')
            
            # Check if the entry has the required fields
            if embedding_file is None or speaker_id is None:
                # For entries without proper embedding info, assign a unique error ID
                entry['unique_speaker_id'] = next_unique_id
                next_unique_id += 1
            else:
                speaker_id = int(speaker_id)
                # The key to look up the cluster ID for this specific segment
                lookup_key = (embedding_file, speaker_id)
                # Assign the pre-calculated cluster ID (now all should have valid IDs)
                entry['unique_speaker_id'] = embedding_to_cluster_id.get(lookup_key, next_unique_id)
                if lookup_key not in embedding_to_cluster_id:
                    next_unique_id += 1
            
            f.write(json.dumps(entry) + '\n')

    print("✅ Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster speaker embeddings from a manifest using DBSCAN and assign unique speaker IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_manifest',
        type=str,
        required=True,
        help='Path to the input JSON manifest file (one JSON object per line).'
    )
    parser.add_argument(
        '--output_manifest',
        type=str,
        required=True,
        help='Path to write the output JSON manifest with the added `unique_speaker_id`.'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.6,
        help='The epsilon value for DBSCAN, defining the neighborhood radius in cosine space.'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=1,
        help='The minimum number of samples for a point to be considered a core point by DBSCAN.'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=256,
        help='The expected feature dimension of the embeddings.'
    )
    parser.add_argument(
        '--no-weighting',
        action='store_true',
        help='Disable utterance-based weighting (use uniform weighting instead).'
    )

    args = parser.parse_args()
    
    cluster_and_update_manifest(
        args.input_manifest,
        args.output_manifest,
        args.eps,
        args.min_samples,
        args.embedding_dim,
        weight_by_utterances=not args.no_weighting
    )