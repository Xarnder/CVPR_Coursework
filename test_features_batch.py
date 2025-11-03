#!/usr/bin/env python3
#
# SCRIPT: test_features_batch.py
#
# PURPOSE: A flexible batch testing script to evaluate any generated descriptor.
#          It runs a set of random queries and saves the visual results to a file.
#          It can generate different visualizations based on the feature type.
#
# EXAMPLE COMMANDS:
#
# 1. To test SPATIAL COLOR features located in a custom directory:
#    python3 test_features_batch.py --descriptor_base_dir ./my_spatial_results/descriptors --descriptor_subdir "spatial_color_4x4_12bins" --feature_type spatial --grid_size 4
#
# 2. To test your OLD GLOBAL 12-bin features (assuming they are in the default './descriptors' folder):
#    python3 test_features_batch.py --descriptor_base_dir ./descriptors --descriptor_subdir "global_rgb_12bins" --feature_type global --color_bins 12
#

import os
import argparse
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm


# --- Helper Functions (unchanged) ---

def cvpr_compare(f1, f2):
    return np.linalg.norm(f1 - f2)


def get_class_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        return int(basename.split('_')[0])
    except (ValueError, IndexError):
        return -1


def create_histogram_visualization(image, bins, vis_height, vis_width):
    hist_canvas = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    channels = cv2.split(image)
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    bin_width = int(np.floor(vis_width / bins))
    if bin_width == 0: bin_width = 1
    for i, (channel, color) in enumerate(zip(channels, colors)):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=vis_height, norm_type=cv2.NORM_MINMAX)
        for j in range(bins):
            x1, y1 = j * bin_width, vis_height
            x2, y2 = (j + 1) * bin_width - 1, vis_height - int(hist[j])
            cv2.rectangle(hist_canvas, (x1, y1), (x2, y2), color, -1)
    return hist_canvas


def create_grid_visualization(image, grid_size):
    h, w, _ = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    image_with_grid = image.copy()
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cv2.rectangle(image_with_grid, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return image_with_grid


# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description='Batch test and visualize search results for a given feature set.')
    # --- NEW ARGUMENT ADDED HERE ---
    parser.add_argument('--descriptor_base_dir', type=str, required=True,
                        help='The base directory containing the descriptor subfolder (e.g., ./my_spatial_results/descriptors).')
    # --- END OF NEW ARGUMENT ---
    parser.add_argument('--descriptor_subdir', type=str, required=True,
                        help='The specific sub-directory of the descriptors to test.')
    parser.add_argument('--feature_type', type=str, required=True, choices=['global', 'spatial'],
                        help="Type of feature being tested.")
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid dimensions, ONLY needed for spatial visualization.')
    parser.add_argument('--color_bins', type=int, default=12,
                        help='Number of color bins, ONLY needed for global visualization.')
    parser.add_argument('--num_queries', type=int, default=20, help='Number of random queries to run.')

    args = parser.parse_args()

    # --- 1. Setup Paths ---
    # --- THIS SECTION IS NOW SIMPLIFIED AND MORE RELIABLE ---
    descriptor_dir = os.path.join(args.descriptor_base_dir, args.descriptor_subdir)

    if not os.path.isdir(descriptor_dir):
        print(f"ERROR: Could not find descriptor directory at the specified path: '{descriptor_dir}'")
        print("Please check both --descriptor_base_dir and --descriptor_subdir arguments.")
        return
    # --- END OF PATHING CHANGES ---

    image_folder = './MSRC_ObjCategImageDatabase_v2/Images'
    results_output_folder = os.path.join('batch_results', args.descriptor_subdir)
    os.makedirs(results_output_folder, exist_ok=True)

    print(f"--- Testing Feature: {args.descriptor_subdir} ---")
    print(f"Loading descriptors from: {descriptor_dir}")
    print(f"Results will be saved in: {results_output_folder}")

    # --- 2. Load Descriptors (unchanged) ---
    print(f"Loading descriptors...")
    ALLFEAT, ALLFILES = [], []
    for filename in os.listdir(descriptor_dir):
        if filename.endswith('.mat'):
            img_path = os.path.join(image_folder, filename.replace(".mat", ".bmp"))
            descriptor_path = os.path.join(descriptor_dir, filename)
            mat_data = sio.loadmat(descriptor_path)
            ALLFILES.append(img_path)
            ALLFEAT.append(mat_data['F'][0])

    ALLFEAT = np.array(ALLFEAT)
    NIMG = len(ALLFILES)
    print(f"Loaded {NIMG} descriptors.")

    # --- 3. Run Batch Queries (unchanged) ---
    for i in tqdm(range(args.num_queries), desc="Running Queries"):
        query_idx = np.random.randint(0, NIMG)
        query_feat = ALLFEAT[query_idx]
        query_filename = os.path.basename(ALLFILES[query_idx])

        dst = [(cvpr_compare(query_feat, feat), i) for i, feat in enumerate(ALLFEAT)]
        dst.sort(key=lambda x: x[0])

        # --- 4. Create Visualization Grid (unchanged) ---
        GRID_ROWS, GRID_COLS = 4, 6
        CELL_WIDTH, CELL_HEIGHT = 320, 250
        IMG_WIDTH, IMG_HEIGHT = 310, 180

        canvas = np.zeros((GRID_ROWS * CELL_HEIGHT, GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)

        for j in range(GRID_ROWS * GRID_COLS):
            dist, result_idx = dst[j]
            img = cv2.imread(ALLFILES[result_idx])
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
            cell[5:5 + IMG_HEIGHT, 5:5 + IMG_WIDTH] = img_resized

            if args.feature_type == 'global':
                vis = create_histogram_visualization(img, args.color_bins, 40, IMG_WIDTH)
                # For global, place vis below the image (not implemented in this version to simplify)
            else:  # spatial
                vis = create_grid_visualization(img_resized, args.grid_size)
                cell[5:5 + IMG_HEIGHT, 5:5 + IMG_WIDTH] = vis

            text_y = IMG_HEIGHT + 25
            font, scale, color = cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255)
            cv2.putText(cell, f"Rank: {j}", (10, text_y), font, scale, color, 1)
            cv2.putText(cell, f"Dist: {dist:.4f}", (100, text_y), font, scale, color, 1)
            cv2.putText(cell, os.path.basename(ALLFILES[result_idx]), (10, text_y + 15), font, scale, color, 1)

            if j == 0:
                cv2.rectangle(cell, (0, 0), (CELL_WIDTH - 1, CELL_HEIGHT - 1), (0, 255, 255), 2)

            row, col = j // GRID_COLS, j % GRID_COLS
            canvas[row * CELL_HEIGHT:(row + 1) * CELL_HEIGHT, col * CELL_WIDTH:(col + 1) * CELL_WIDTH] = cell

        # --- 5. Save the final canvas (unchanged) ---
        out_path = os.path.join(results_output_folder, f"query_{query_filename.replace('.bmp', '.png')}")
        cv2.imwrite(out_path, canvas)

    print(f"\n--- Batch test complete for {args.descriptor_subdir}! ---")


if __name__ == '__main__':
    main()