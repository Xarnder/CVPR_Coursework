#!/usr/bin/env python3
#
# SCRIPT: compute_pca_mahalanobis.py
#
# PURPOSE: Implements Requirement 4. This script loads a set of high-dimensional
#          descriptors, applies PCA to reduce their dimensionality, and then
#          evaluates visual search performance comparing Euclidean distance
#          and Mahalanobis distance in the reduced PCA space.
#          It generates multiple visualizations for analysis and reporting.
#
# EXAMPLE COMMAND:
#
# python3 compute_pca_mahalanobis.py \
#   --descriptor_dir ./descriptors/global_rgb_12bins \
#   --output_dir ./pca_mahalanobis_results_12bins_32d \
#   --n_components 32 \
#   --num_queries 10
#

import os
import argparse
import shutil
import re
from typing import Dict, List, Tuple

import numpy as np
import cv2
import scipy.io as sio
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration for Visualization ---
CELL_WIDTH, CELL_HEIGHT = 220, 200
IMG_WIDTH, IMG_HEIGHT = 200, 120
GRID_COLS = 11  # 1 Query image + 10 results
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (255, 255, 255)


# --- Helper Functions ---

def ensure_dir(path: str):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        print(f"DEBUG: Creating output directory: {path}")
        os.makedirs(path)


def get_class_from_filename(filename: str) -> str:
    """Extracts the class name (e.g., 'tree', 'car') from the filename stem."""
    name = os.path.basename(filename)
    stem = os.path.splitext(name)[0]
    # Use regex to find the first sequence of letters after the numbers
    match = re.search(r'^\d+_\d+_(.+)$', stem)
    if match:
        return match.group(1)
    # Fallback for filenames like '1_1_s.bmp'
    return stem.split('_', 1)[0]


def load_descriptors(descriptor_dir: str, dataset_images_path: str) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Loads all .mat descriptors from a directory."""
    if not os.path.isdir(descriptor_dir):
        print(f"FATAL: Descriptor directory not found at '{descriptor_dir}'. Please check the path.")
        exit(1)

    features, file_paths, file_map = [], [], {}
    print(f"Loading descriptors from '{descriptor_dir}'...")
    for filename in tqdm(os.listdir(descriptor_dir), desc="Loading .mat files"):
        if not filename.endswith('.mat'):
            continue

        mat_path = os.path.join(descriptor_dir, filename)
        image_filename = filename.replace('.mat', '.bmp')
        image_path = os.path.join(dataset_images_path, image_filename)

        if not os.path.exists(image_path):
            print(f"DEBUG: Skipping descriptor '{filename}' as corresponding image not found at '{image_path}'")
            continue

        try:
            mat_data = sio.loadmat(mat_path)
            # Ensure the feature vector is flattened to 1D
            feature_vector = mat_data['F'].ravel()
            features.append(feature_vector)
            file_paths.append(image_path)
            file_map[image_path] = get_class_from_filename(image_path)
        except Exception as e:
            print(f"DEBUG: Error loading or processing '{filename}': {e}")

    if not features:
        print(f"FATAL: No descriptors were successfully loaded from '{descriptor_dir}'.")
        exit(1)

    return np.array(features), file_paths, file_map


def create_result_cell(rank: int, img_path: str, distance: float, is_query: bool = False) -> np.ndarray:
    """Creates a single visual cell for the results grid."""
    cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    try:
        img = cv2.imread(img_path)
        if img is None: raise IOError("Image is None")
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    except Exception as e:
        print(f"DEBUG: Could not read/resize image {img_path}: {e}")
        img_resized = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 40, dtype=np.uint8)
        cv2.putText(img_resized, "Error", (10, IMG_HEIGHT // 2), FONT, 1, (0, 0, 255), 2)

    # Place image in cell
    y_start, x_start = 5, 10
    cell[y_start:y_start + IMG_HEIGHT, x_start:x_start + IMG_WIDTH] = img_resized

    # Add text info
    text_y_base = y_start + IMG_HEIGHT + 20
    filename_text = os.path.basename(img_path)

    if is_query:
        cv2.putText(cell, "QUERY", (x_start, text_y_base), FONT, FONT_SCALE, (0, 255, 255), 1)
        cv2.putText(cell, filename_text, (x_start, text_y_base + 15), FONT, FONT_SCALE, FONT_COLOR, 1)
        cv2.rectangle(cell, (0, 0), (CELL_WIDTH - 1, CELL_HEIGHT - 1), (0, 255, 255), 3)
    else:
        rank_text = f"Rank: {rank}"
        dist_text = f"Dist: {distance:.4f}"
        cv2.putText(cell, rank_text, (x_start, text_y_base), FONT, FONT_SCALE, FONT_COLOR, 1)
        cv2.putText(cell, dist_text, (x_start + 80, text_y_base), FONT, FONT_SCALE, FONT_COLOR, 1)
        cv2.putText(cell, filename_text, (x_start, text_y_base + 15), FONT, FONT_SCALE, FONT_COLOR, 1)

    return cell


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description='Run PCA and compare Euclidean vs. Mahalanobis distance.')
    parser.add_argument('--descriptor_dir', type=str, required=True,
                        help='Path to the directory of pre-computed descriptors (e.g., ./descriptors/global_rgb_12bins).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save all generated plots and result images.')
    parser.add_argument('--n_components', type=int, default=32,
                        help='Number of principal components to keep.')
    parser.add_argument('--num_queries', type=int, default=10,
                        help='Number of random query images to test.')
    args = parser.parse_args()

    # --- 1. Setup ---
    print("--- Starting PCA & Mahalanobis Distance Evaluation ---")
    print(f"Config: Using {args.n_components} PCA components, running {args.num_queries} queries.")
    print("NOTE: scikit-learn's PCA and numpy's linear algebra run on the CPU. A GPU is not used by this script.")

    ensure_dir(args.output_dir)
    dataset_images_path = './MSRC_ObjCategImageDatabase_v2/Images'

    # --- 2. Load Data ---
    all_features, all_files, class_map = load_descriptors(args.descriptor_dir, dataset_images_path)
    print(f"Successfully loaded {all_features.shape[0]} descriptors of dimension {all_features.shape[1]}.")

    # --- 3. PCA: Train and Transform ---
    print(f"\n--- Performing PCA to reduce dimension to {args.n_components} ---")
    pca = PCA(n_components=args.n_components)

    # Fit PCA on the entire dataset and transform it
    pca_features = pca.fit_transform(all_features)
    print(f"Data transformed to new shape: {pca_features.shape}")

    # --- 4. PCA Visualizations (for your report) ---

    # a) Explained Variance Plot
    plt.figure(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    plt.plot(range(1, args.n_components + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.ylim(0, 1.05)
    variance_path = os.path.join(args.output_dir, 'pca_explained_variance.png')
    plt.savefig(variance_path)
    plt.close()
    print(f"Saved explained variance plot to: {variance_path}")
    print(f"Total variance captured by {args.n_components} components: {cumulative_variance[-1]:.4f}")

    # b) 2D Scatter Plot of Projected Data
    if args.n_components >= 2:
        unique_classes = sorted(list(set(class_map.values())))
        colors = plt.cm.get_cmap('tab20', len(unique_classes))

        plt.figure(figsize=(12, 8))
        for i, class_name in enumerate(unique_classes):
            indices = [idx for idx, path in enumerate(all_files) if class_map[path] == class_name]
            plt.scatter(pca_features[indices, 0], pca_features[indices, 1],
                        label=class_name, color=colors(i), alpha=0.7)
        plt.title('Data Projected onto First 2 Principal Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        scatter_path = os.path.join(args.output_dir, 'pca_2d_scatter.png')
        plt.savefig(scatter_path)
        plt.close()
        print(f"Saved 2D scatter plot to: {scatter_path}")

    # --- 5. Prepare for Mahalanobis Distance ---
    print("\n--- Preparing for Mahalanobis Distance Calculation ---")
    print("Calculating the inverse covariance matrix of the PCA-transformed data...")
    try:
        cov_matrix = np.cov(pca_features, rowvar=False)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        print("Successfully computed the inverse covariance matrix.")
    except np.linalg.LinAlgError as e:
        print(f"FATAL: Could not compute inverse covariance matrix: {e}")
        print("This can happen if features are perfectly correlated. Try a different descriptor or n_components.")
        exit(1)

    # --- 6. Run Batch Queries and Compare Distances ---
    print(f"\n--- Running {args.num_queries} Random Queries for Comparison ---")
    n_images = len(all_files)

    for i in tqdm(range(args.num_queries), desc="Processing Queries"):
        query_idx = np.random.randint(0, n_images)
        query_pca_feat = pca_features[query_idx]
        query_img_path = all_files[query_idx]
        query_filename_stem = os.path.splitext(os.path.basename(query_img_path))[0]

        # a) Euclidean Search
        euclidean_dists = np.linalg.norm(pca_features - query_pca_feat, axis=1)
        euclidean_ranked_indices = np.argsort(euclidean_dists)

        # b) Mahalanobis Search
        mahalanobis_dists = np.array([mahalanobis(query_pca_feat, f, inv_cov_matrix) for f in pca_features])
        mahalanobis_ranked_indices = np.argsort(mahalanobis_dists)

        # c) Create Comparison Visualization
        # ##################################################################
        # #######                THE FIX IS HERE                     #######
        # ##################################################################
        canvas = np.zeros((2 * CELL_HEIGHT + 80, GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)

        # Add titles for each row
        cv2.putText(canvas, "Euclidean Distance Results (in PCA space)", (10, 25), FONT, 0.8, (255, 255, 0), 2)
        cv2.putText(canvas, "Mahalanobis Distance Results (in PCA space)", (10, CELL_HEIGHT + 55), FONT, 0.8,
                    (255, 255, 0), 2)

        # Create query cell once
        query_cell = create_result_cell(0, query_img_path, 0.0, is_query=True)

        # Populate Euclidean row
        y_start_euclidean = 40
        canvas[y_start_euclidean: y_start_euclidean + CELL_HEIGHT, 0:CELL_WIDTH] = query_cell
        for rank in range(1, GRID_COLS):
            result_idx = euclidean_ranked_indices[rank]
            dist = euclidean_dists[result_idx]
            cell = create_result_cell(rank, all_files[result_idx], dist)
            x_start = rank * CELL_WIDTH
            canvas[y_start_euclidean:y_start_euclidean + CELL_HEIGHT, x_start:x_start + CELL_WIDTH] = cell

        # Populate Mahalanobis row
        y_start_mahalanobis = y_start_euclidean + CELL_HEIGHT + 30
        canvas[y_start_mahalanobis:y_start_mahalanobis + CELL_HEIGHT, 0:CELL_WIDTH] = query_cell
        for rank in range(1, GRID_COLS):
            result_idx = mahalanobis_ranked_indices[rank]
            dist = mahalanobis_dists[result_idx]
            cell = create_result_cell(rank, all_files[result_idx], dist)
            x_start = rank * CELL_WIDTH
            canvas[y_start_mahalanobis:y_start_mahalanobis + CELL_HEIGHT, x_start:x_start + CELL_WIDTH] = cell

        # Save the final canvas
        output_path = os.path.join(args.output_dir, f"query_{i + 1}_{query_filename_stem}.png")
        cv2.imwrite(output_path, canvas)

    print("\n--- All tasks complete! ---")
    print(f"Check the '{args.output_dir}' directory for all generated analysis plots and query results.")


if __name__ == '__main__':
    main()