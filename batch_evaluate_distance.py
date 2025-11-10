#!/usr/bin/env python3
#
# SCRIPT: batch_evaluate_distance.py
#
# PURPOSE: Implements a batch version of distance evaluation (Requirement 5).
#          It systematically tests all descriptor sets found in a base directory
#          against multiple distance metrics. It uses a large random set of queries
#          for robust metric calculation and a smaller, FIXED subset of those queries
#          for generating directly comparable visual result grids.
#
# EXAMPLE COMMAND:
# python3 batch_evaluate_distance.py \
#   --base_descriptor_dir ./descriptors \
#   --distance_metrics l1 l2 chi2 cosine \
#   --output_base_dir ./batch_distance_results \
#   --num_queries 100 \
#   --num_visual_queries 5 \
#   --top_k 10
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

# --- Configuration & Distance Function Definitions ---

# Metrics available for testing
DISTANCE_FUNCS = {
    'l1': lambda f1, f2: np.sum(np.abs(f1 - f2)),  # L1 Norm
    'l2': lambda f1, f2: np.linalg.norm(f1 - f2),  # L2 Norm / Euclidean
    'cosine': cosine,  # Cosine Distance
    'chi2': lambda f1, f2: 0.5 * np.sum(((f1 - f2) ** 2) / (f1 + f2 + 1e-10))  # Chi-Squared
}

# Visualization configuration
DATASET_IMAGES_PATH = './MSRC_ObjCategImageDatabase_v2/Images'
CELL_WIDTH, CELL_HEIGHT = 220, 180
IMG_WIDTH, IMG_HEIGHT = 200, 120
GRID_ROWS, GRID_COLS = 4, 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255, 255, 255)


def get_class_from_filename(filename: str) -> str:
    """Extracts the class ID from the filename (e.g., '1' from '1_1_s.bmp')."""
    basename = os.path.basename(filename)
    try:
        # Assumes format like '1_12_image.bmp'
        return basename.split('_')[0]
    except (ValueError, IndexError):
        # Fallback for filenames without underscores if needed
        return "unknown"


def load_descriptors_and_classes(descriptor_dir: str):
    """Loads features, file paths, and class labels from a single directory."""
    all_features, all_files, all_classes = [], [], []

    if not os.path.isdir(descriptor_dir):
        print(f"ERROR: Descriptor directory not found at '{descriptor_dir}'. Skipping.")
        return None, None, None

    for filename in sorted(os.listdir(descriptor_dir)):
        if filename.endswith('.mat'):
            mat_path = os.path.join(descriptor_dir, filename)
            img_path = os.path.join(DATASET_IMAGES_PATH, filename.replace('.mat', '.bmp'))
            if os.path.exists(img_path):
                try:
                    mat_data = sio.loadmat(mat_path)
                    all_features.append(mat_data['F'].ravel())
                    all_files.append(img_path)
                    all_classes.append(get_class_from_filename(filename))
                except Exception as e:
                    print(f"Warning: Error loading {mat_path}: {e}")

    if not all_features:
        print(f"Warning: No valid descriptors loaded from {descriptor_dir}.")
        return None, None, None

    return np.array(all_features), np.array(all_files), np.array(all_classes)


def create_visual_grid(distances, all_files, query_idx):
    """Generates the visual grid for a single search result."""
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

        cv2.putText(cell, f"Rank: {j}", (x_start, text_y_base), FONT, 0.4, FONT_COLOR, 1)
        cv2.putText(cell, f"Dist: {dist:.4f}", (x_start + 70, text_y_base), FONT, 0.4, FONT_COLOR, 1)
        cv2.putText(cell, os.path.basename(img_path), (x_start, text_y_base + 15), FONT, 0.4, FONT_COLOR, 1)

        if j == 0:  # Highlight the query image
            cv2.rectangle(cell, (0, 0), (CELL_WIDTH - 1, CELL_HEIGHT - 1), (0, 255, 255), 2)

        row, col = j // GRID_COLS, j % GRID_COLS
        canvas[row * CELL_HEIGHT:(row + 1) * CELL_HEIGHT, col * CELL_WIDTH:(col + 1) * CELL_WIDTH] = cell
    return canvas


def run_search_and_log(descriptor_name: str, all_features: np.ndarray, all_files: np.ndarray, all_classes: np.ndarray,
                       metric_name: str, dist_func, query_indices, args):
    """Performs the search, calculates metrics, generates visualization for a subset, and logs results."""

    metric_output_dir = os.path.join(args.output_base_dir, descriptor_name, f'{metric_name}_results')
    os.makedirs(metric_output_dir, exist_ok=True)

    query_precisions, query_recalls = [], []

    # Iterate through ALL selected queries for METRIC calculation
    for i, query_idx in enumerate(tqdm(query_indices, desc=f" {descriptor_name} / {metric_name.upper()}")):
        query_feat = all_features[query_idx]
        query_class = all_classes[query_idx]

        distances = [(dist_func(query_feat, feat), idx) for idx, feat in enumerate(all_features)]
        distances.sort(key=lambda x: x[0])

        # --- METRIC CALCULATION ---
        top_k_indices = [idx for dist, idx in distances[1:args.top_k + 1]]
        top_k_classes = all_classes[top_k_indices]

        true_positives = np.sum(top_k_classes == query_class)
        precision_at_k = true_positives / args.top_k
        query_precisions.append(precision_at_k)

        total_relevant_in_dataset = np.sum(all_classes == query_class) - 1
        recall_at_k = true_positives / total_relevant_in_dataset if total_relevant_in_dataset > 0 else 0.0
        query_recalls.append(recall_at_k)

        # --- VISUALIZATION GRID CREATION (ONLY FOR THE FIRST N QUERIES) ---
        if i < args.num_visual_queries:
            canvas = create_visual_grid(distances, all_files, query_idx)
            query_img_path = all_files[query_idx]
            query_filename_stem = os.path.splitext(os.path.basename(query_img_path))[0]
            output_path = os.path.join(metric_output_dir, f"query_{query_filename_stem}.png")
            cv2.imwrite(output_path, canvas)

    # --- AVERAGE METRICS AND WRITE TO CSV ---
    mean_precision = np.mean(query_precisions)
    mean_recall = np.mean(query_recalls)

    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'descriptor_name': descriptor_name,
        'distance_metric': metric_name,
        'num_queries_for_metrics': args.num_queries,
        'seed': args.seed,
        'top_k': args.top_k,
        f'mean_precision_at_{args.top_k}': f"{mean_precision:.4f}",
        f'mean_recall_at_{args.top_k}': f"{mean_recall:.4f}"
    }

    file_exists = os.path.isfile(args.master_csv_path)
    with open(args.master_csv_path, 'a', newline='') as csvfile:
        fieldnames = log_entry.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    print(f"  -> P@{args.top_k}: {mean_precision:.4f}, R@{args.top_k}: {mean_recall:.4f}. Logged to CSV.")


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate multiple descriptor sets against various distance metrics.')
    parser.add_argument('--base_descriptor_dir', type=str, default='./descriptors',
                        help='Base directory containing all descriptor sub-folders (e.g., ./descriptors).')
    parser.add_argument('--output_base_dir', type=str, default='./batch_distance_results',
                        help='Base directory to save all result grids, grouped by descriptor and metric.')
    parser.add_argument('--distance_metrics', nargs='+', default=['l1', 'l2', 'chi2', 'cosine'],
                        choices=list(DISTANCE_FUNCS.keys()),
                        help='A list of distance metrics to test.')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of random queries to run FOR METRICS.')
    parser.add_argument('--num_visual_queries', type=int, default=5, help='Number of FIXED queries for VISUALIZATION.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for selecting all query images (default 123).')
    parser.add_argument('--top_k', type=int, default=10, help='Value of K for Precision@K and Recall@K metrics.')
    parser.add_argument('--master_csv_path', type=str, default='./master_distance_log.csv',
                        help='Path to the master CSV file for logging all experiment results.')
    args = parser.parse_args()

    if args.num_visual_queries > args.num_queries:
        print(
            f"Warning: num_visual_queries ({args.num_visual_queries}) cannot be greater than num_queries ({args.num_queries}).")
        args.num_visual_queries = args.num_queries
        print(f"Setting num_visual_queries to {args.num_visual_queries}.")

    print(f"--- Starting Batch Distance Evaluation ---")
    print(f"Base Descriptor Directory: {args.base_descriptor_dir}")
    print(f"Metrics to Test: {args.distance_metrics}")
    print(f"Using seed: {args.seed} to select {args.num_queries} queries for metrics.")
    print(f"The first {args.num_visual_queries} of these will be saved as visual examples.")

    # 1. Discover Descriptor Folders
    descriptor_dirs = [
        d for d in os.listdir(args.base_descriptor_dir)
        if os.path.isdir(os.path.join(args.base_descriptor_dir, d))
    ]

    if not descriptor_dirs:
        print(f"FATAL: No descriptor sub-directories found in '{args.base_descriptor_dir}'. Exiting.")
        return

    descriptor_dirs.sort()
    print(f"Found {len(descriptor_dirs)} descriptor sets to evaluate: {descriptor_dirs}")
    print("-" * 40)

    # 2. Main Batch Loop
    for descriptor_name in descriptor_dirs:
        full_descriptor_path = os.path.join(args.base_descriptor_dir, descriptor_name)

        print(f"\n[DESCRIPTOR SET: {descriptor_name}]")

        # Load data once per descriptor set
        all_features, all_files, all_classes = load_descriptors_and_classes(full_descriptor_path)

        if all_features is None:
            continue

        print(f"  - Loaded {len(all_files)} images. Dimension: {all_features.shape[1]}")

        # **IMPORTANT**: Generate the fixed list of query indices ONCE per descriptor set
        np.random.seed(args.seed)
        n_images = len(all_files)
        # This list of indices will now be the same for every distance metric tested on this descriptor
        fixed_query_indices = np.random.choice(n_images, size=args.num_queries, replace=False)

        # Loop through all distance metrics for the current descriptor
        for metric_name in args.distance_metrics:
            dist_func = DISTANCE_FUNCS[metric_name]

            # Pass the fixed list of queries to the evaluation function
            run_search_and_log(
                descriptor_name,
                all_features,
                all_files,
                all_classes,
                metric_name,
                dist_func,
                fixed_query_indices,  # Use the same list of queries
                args
            )

    print("\n--- All Batch Evaluations Complete! ---")
    print(f"Results logged in: {args.master_csv_path}")


if __name__ == '__main__':
    main()