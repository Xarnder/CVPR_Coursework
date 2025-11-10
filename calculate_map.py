#!/usr/bin/env python3
#
# SCRIPT: calculate_map.py
#
# PURPOSE: A dedicated script to calculate the Mean Average Precision (mAP) for
#          all descriptor sets found in a given directory. This provides a single,
#          robust metric for ranking the overall performance of each descriptor.
#
#          The script iterates through every image as a query, calculates its
#          Average Precision (AP), and then averages these scores to get the mAP.
#
# USAGE:
#   python3 calculate_map.py --base_descriptor_dir ./descriptors
#

import os
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# ---- Configuration ----
DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'


# ---- Helper Functions (adapted from evaluate_system.py) ----

def get_class_from_filename(path_or_name: str) -> str:
    """Extracts the class name (e.g., '1' for cars) from a filename."""
    name = os.path.basename(path_or_name)
    stem = os.path.splitext(name)[0]
    return stem.split('_', 1)[0]


def load_experiment(base_dir: str, descriptor_subdir: str):
    """Loads all features and class labels for a given experiment."""
    desc_dir = os.path.join(base_dir, descriptor_subdir)
    if not os.path.isdir(desc_dir):
        raise FileNotFoundError(f"Descriptor directory not found: {desc_dir}")

    feats, files, classes = [], [], []
    for filename in sorted(os.listdir(desc_dir)):
        if not filename.endswith('.mat'):
            continue

        img_path = os.path.join(DATASET_FOLDER, 'Images', filename.replace('.mat', '.bmp'))
        if not os.path.exists(img_path):
            continue

        try:
            mat = sio.loadmat(os.path.join(desc_dir, filename))
            if 'F' in mat:
                F = mat['F'].ravel().astype(np.float64)
                feats.append(F)
                files.append(img_path)
                classes.append(get_class_from_filename(img_path))
        except Exception as e:
            print(f"Warning: Could not load {filename}. Error: {e}")

    if not files:
        raise RuntimeError(f"No valid .mat descriptors found in {desc_dir}")

    return np.vstack(feats), list(files), np.array(classes)


def rank_for_query(query_idx: int, all_features: np.ndarray):
    """Ranks all images by Euclidean distance to a query feature."""
    query_feature = all_features[query_idx]
    # Calculate L2 norm (Euclidean distance) between the query and all other features
    distances = np.linalg.norm(all_features - query_feature, axis=1)

    # Get the indices that would sort the distances array
    ranked_indices = np.argsort(distances)

    # Return ranked indices, excluding the query image itself
    return ranked_indices[ranked_indices != query_idx]


def calculate_ap_for_query(query_idx: int, all_features: np.ndarray, all_classes: np.ndarray) -> float:
    """
    Calculates the Average Precision (AP) for a single query.
    AP is the average of precision values obtained at each relevant item in the ranked list.
    """
    true_class = all_classes[query_idx]

    # Calculate the total number of relevant images in the dataset (excluding the query)
    total_relevant = np.sum(all_classes == true_class) - 1
    if total_relevant <= 0:
        return 0.0  # Cannot calculate AP if there are no other relevant images

    # Get the ranked list of image indices for the query
    ranked_indices = rank_for_query(query_idx, all_features)
    ranked_classes = all_classes[ranked_indices]

    # Calculate precision at each relevant position
    true_positives = 0
    precisions_at_k = []

    for i, retrieved_class in enumerate(ranked_classes):
        rank = i + 1
        if retrieved_class == true_class:
            true_positives += 1
            precision = true_positives / rank
            precisions_at_k.append(precision)

    if not precisions_at_k:
        return 0.0  # No relevant items were retrieved

    # Average Precision is the mean of precisions at each correct retrieval
    average_precision = np.mean(precisions_at_k)
    return average_precision


def main():
    parser = argparse.ArgumentParser(description="Calculate Mean Average Precision (mAP) for all descriptor sets.")
    parser.add_argument('--base_descriptor_dir', type=str, default='./descriptors',
                        help="The base directory containing descriptor sub-folders.")
    args = parser.parse_args()

    # Discover all subdirectories in the base descriptor directory
    if not os.path.isdir(args.base_descriptor_dir):
        print(f"FATAL: Base descriptor directory not found at '{args.base_descriptor_dir}'")
        return

    experiment_folders = [name for name in os.listdir(args.base_descriptor_dir)
                          if os.path.isdir(os.path.join(args.base_descriptor_dir, name))]
    experiment_folders.sort()

    if not experiment_folders:
        print(f"FATAL: No descriptor sub-directories found in '{args.base_descriptor_dir}'.")
        return

    print(f"Found {len(experiment_folders)} experiments to evaluate for mAP.")

    results = []

    # Main loop to process each descriptor set
    for subfolder in experiment_folders:
        print(f"\n--- Processing experiment: {subfolder} ---")
        try:
            features, files, classes = load_experiment(args.base_descriptor_dir, subfolder)
            num_images = len(files)

            all_aps = []
            # Iterate through every image as a query
            for i in tqdm(range(num_images), desc="Calculating AP for each query"):
                ap = calculate_ap_for_query(i, features, classes)
                all_aps.append(ap)

            # Mean Average Precision is the mean of all individual APs
            mean_ap = np.mean(all_aps)
            results.append({'experiment': subfolder, 'mAP': mean_ap})
            print(f"mAP for '{subfolder}': {mean_ap:.4f}")

        except (FileNotFoundError, RuntimeError) as e:
            print(f"Could not process '{subfolder}': {e}")

    # --- Final Summary ---
    if not results:
        print("\nNo results were calculated.")
        return

    # Sort results by mAP score in descending order
    results.sort(key=lambda x: x['mAP'], reverse=True)

    print("\n\n--- Final mAP Results (Ranked by Performance) ---")
    print("=" * 50)
    print(f"{'Descriptor Set':<35} | {'mAP':<10}")
    print("-" * 50)
    for res in results:
        print(f"{res['experiment']:<35} | {res['mAP']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()