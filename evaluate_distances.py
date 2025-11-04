#!/usr/bin/env python3
#
# SCRIPT: evaluate_distances.py
#
# PURPOSE: Implements Requirement 5. This script tests a given descriptor set
#          against multiple distance metrics, generates visual results, AND
#          calculates performance metrics (Precision@K, Recall@K), appending
#          them to a master CSV log file for systematic analysis.
#
# EXAMPLE COMMANDS:
#
# python3 evaluate_distances.py \
#   --descriptor_dir ./descriptors/global_rgb_12bins \
#   --output_dir ./distance_results/global_12bins \
#   --distance_metrics l1 l2 chi2 \
#   --seed 123 \
#   --top_k 10 \
#   --master_csv_path ./master_evaluation_log.csv
#

import os
import argparse
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cosine
import cv2
from tqdm import tqdm
import csv
from datetime import datetime


# --- Distance Function Definitions (unchanged) ---

def l1_dist(f1, f2):
    """L1 Norm / Manhattan Distance"""
    return np.sum(np.abs(f1 - f2))


def l2_dist(f1, f2):
    """L2 Norm / Euclidean Distance"""
    return np.linalg.norm(f1 - f2)


def cosine_dist(f1, f2):
    """Cosine Distance (1 - Cosine Similarity)"""
    return cosine(f1, f2)


def chi2_dist(f1, f2):
    """Chi-Squared Distance"""
    eps = 1e-10
    return 0.5 * np.sum(((f1 - f2) ** 2) / (f1 + f2 + eps))


DISTANCE_FUNCS = {
    'l1': l1_dist, 'l2': l2_dist,
    'cosine': cosine_dist, 'chi2': chi2_dist
}

# --- Visualization & Helper Functions ---

CELL_WIDTH, CELL_HEIGHT = 220, 180
IMG_WIDTH, IMG_HEIGHT = 200, 120
GRID_ROWS, GRID_COLS = 4, 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_class_from_filename(filename):
    """Extracts the class ID from the filename (e.g., '1' from '1_1_s.bmp')."""
    basename = os.path.basename(filename)
    try:
        return basename.split('_')[0]
    except (ValueError, IndexError):
        return "unknown"


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate descriptor performance with various distance metrics and log results.')
    parser.add_argument('--descriptor_dir', type=str, required=True,
                        help='Path to the directory of pre-computed descriptors to test.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory to save all result grids.')
    parser.add_argument('--distance_metrics', nargs='+', required=True, choices=list(DISTANCE_FUNCS.keys()),
                        help='A list of distance metrics to test.')
    parser.add_argument('--num_queries', type=int, default=10, help='Number of random queries to run.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for selecting query images.')
    # --- NEW ARGUMENTS FOR METRICS AND LOGGING ---
    parser.add_argument('--top_k', type=int, default=10, help='Value of K for Precision@K and Recall@K metrics.')
    parser.add_argument('--master_csv_path', type=str, default='./master_evaluation_log.csv',
                        help='Path to the master CSV file for logging all experiment results.')
    args = parser.parse_args()

    # --- 1. Load Descriptors and Class Labels ---
    print(f"--- Loading data from: {args.descriptor_dir} ---")
    image_folder = './MSRC_ObjCategImageDatabase_v2/Images'
    all_features, all_files, all_classes = [], [], []

    if not os.path.isdir(args.descriptor_dir):
        print(f"FATAL: Descriptor directory not found at '{args.descriptor_dir}'")
        return

    for filename in sorted(os.listdir(args.descriptor_dir)):
        if filename.endswith('.mat'):
            mat_path = os.path.join(args.descriptor_dir, filename)
            img_path = os.path.join(image_folder, filename.replace('.mat', '.bmp'))
            if os.path.exists(img_path):
                mat_data = sio.loadmat(mat_path)
                all_features.append(mat_data['F'].ravel())
                all_files.append(img_path)
                all_classes.append(get_class_from_filename(filename))

    all_features = np.array(all_features)
    all_classes = np.array(all_classes)
    n_images = len(all_files)
    print(f"Loaded {n_images} descriptors of dimension {all_features.shape[1]}.")

    # --- 2. Pre-select Query Images for Reproducibility ---
    np.random.seed(args.seed)
    query_indices = np.random.choice(n_images, size=args.num_queries, replace=False)
    print(f"Using seed {args.seed} to select {len(query_indices)} fixed queries for all tests.")

    # --- 3. Loop Through Metrics, Run Queries, and Calculate Metrics ---
    for metric_name in args.distance_metrics:
        print(f"\n--- Evaluating with '{metric_name.upper()}' distance ---")
        dist_func = DISTANCE_FUNCS[metric_name]

        metric_output_dir = os.path.join(args.output_dir, f'{metric_name}_results')
        os.makedirs(metric_output_dir, exist_ok=True)

        # --- NEW: Lists to store metrics for each query ---
        query_precisions = []
        query_recalls = []

        for query_idx in tqdm(query_indices, desc=f"Queries ({metric_name})"):
            query_feat = all_features[query_idx]
            query_class = all_classes[query_idx]

            distances = [(dist_func(query_feat, feat), idx) for idx, feat in enumerate(all_features)]
            distances.sort(key=lambda x: x[0])

            # --- NEW: METRIC CALCULATION FOR THIS QUERY ---
            # Get top K results (excluding the query itself at index 0)
            top_k_indices = [idx for dist, idx in distances[1:args.top_k + 1]]
            top_k_classes = all_classes[top_k_indices]

            # Calculate True Positives
            true_positives = np.sum(top_k_classes == query_class)

            # Calculate Precision@K
            precision_at_k = true_positives / args.top_k
            query_precisions.append(precision_at_k)

            # Calculate Recall@K
            total_relevant_in_dataset = np.sum(all_classes == query_class) - 1  # Exclude query image itself
            if total_relevant_in_dataset > 0:
                recall_at_k = true_positives / total_relevant_in_dataset
            else:
                recall_at_k = 0.0  # Avoid division by zero
            query_recalls.append(recall_at_k)

            # --- Visualization Grid creation (unchanged) ---
            canvas = np.zeros((GRID_ROWS * CELL_HEIGHT, GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)
            for j in range(min(len(distances), GRID_ROWS * GRID_COLS)):
                dist, result_idx = distances[j]
                img_path = all_files[result_idx]
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) if img is not None else np.zeros(
                    (IMG_HEIGHT, IMG_WIDTH, 3))
                cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
                y_start, x_start = 5, 10
                cell[y_start:y_start + IMG_HEIGHT, x_start:x_start + IMG_WIDTH] = img_resized
                text_y_base = y_start + IMG_HEIGHT + 20
                cv2.putText(cell, f"Rank: {j}", (x_start, text_y_base), FONT, 0.4, (255, 255, 255), 1)
                cv2.putText(cell, f"Dist: {dist:.4f}", (x_start + 70, text_y_base), FONT, 0.4, (255, 255, 255), 1)
                cv2.putText(cell, os.path.basename(img_path), (x_start, text_y_base + 15), FONT, 0.4, (255, 255, 255),
                            1)
                if j == 0:
                    cv2.rectangle(cell, (0, 0), (CELL_WIDTH - 1, CELL_HEIGHT - 1), (0, 255, 255), 2)
                row, col = j // GRID_COLS, j % GRID_COLS
                canvas[row * CELL_HEIGHT:(row + 1) * CELL_HEIGHT, col * CELL_WIDTH:(col + 1) * CELL_WIDTH] = cell

            query_img_path = all_files[query_idx]
            query_filename_stem = os.path.splitext(os.path.basename(query_img_path))[0]
            output_path = os.path.join(metric_output_dir, f"query_{query_filename_stem}.png")
            cv2.imwrite(output_path, canvas)

        # --- NEW: AVERAGE METRICS AND WRITE TO CSV ---
        mean_precision = np.mean(query_precisions)
        mean_recall = np.mean(query_recalls)

        print(f"  - Mean Precision@{args.top_k}: {mean_precision:.4f}")
        print(f"  - Mean Recall@{args.top_k}: {mean_recall:.4f}")

        # Prepare data for the CSV row
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'descriptor_name': os.path.basename(args.descriptor_dir),
            'distance_metric': metric_name,
            'num_queries': args.num_queries,
            'seed': args.seed,
            'top_k': args.top_k,
            f'mean_precision_at_{args.top_k}': f"{mean_precision:.4f}",
            f'mean_recall_at_{args.top_k}': f"{mean_recall:.4f}"
        }

        # Write to the master CSV file
        file_exists = os.path.isfile(args.master_csv_path)
        with open(args.master_csv_path, 'a', newline='') as csvfile:
            fieldnames = log_entry.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Write header only if file is new
            writer.writerow(log_entry)

        print(f"  - Results appended to '{args.master_csv_path}'")

    print("\n--- All evaluations complete! ---")


if __name__ == '__main__':
    main()