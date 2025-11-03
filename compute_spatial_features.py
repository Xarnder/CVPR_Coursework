#!/usr/bin/env python3
# (All comments and examples are the same)

import os
import argparse
import shutil
import cv2
import numpy as np
import scipy.io as sio
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_feature_descriptor(image, feature_type, color_bins, lbp_points, lbp_radius):
    """Computes a histogram for a single image patch (cell)."""
    if feature_type == 'color':
        img_8bit = (image * 255).astype(np.uint8)
        hist = cv2.calcHist([img_8bit], [0, 1, 2], None,
                            [color_bins, color_bins, color_bins],
                            [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist.flatten()

    elif feature_type == 'texture':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # THE CRUCIAL FIX IS HERE:
        # The number of bins for a 'uniform' LBP with P points is ALWAYS P + 2.
        # We must use this fixed, constant value.
        n_bins = lbp_points + 2

        lbp = local_binary_pattern(gray, P=lbp_points, R=lbp_radius, method='uniform')

        # Create a histogram with the FIXED number of bins and range.
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

        # Normalize the histogram manually to be safe.
        hist = hist.astype("float")
        eps = 1e-7  # A small number to prevent division by zero for empty cells
        hist /= (hist.sum() + eps)

        return hist

    else:
        raise ValueError("Unknown feature_type specified.")


def main():
    # ... (The rest of the main function is IDENTICAL to the previous version)
    # ... It already contains the fix for empty cells, which is also needed.
    # ... No need to copy it all here again, just replace the function above.
    parser = argparse.ArgumentParser(description='Compute spatial grid features for an image dataset.')
    parser.add_argument('--dataset_dir', type=str, default='./MSRC_ObjCategImageDatabase_v2',
                        help='Path to the root of the MSRC dataset.')
    parser.add_argument('--output_dir', type=str, default='./spatial_grid_outputs',
                        help='Path to the directory where all outputs will be saved.')
    parser.add_argument('--feature_type', type=str, required=True, choices=['color', 'texture'],
                        help="Type of feature to extract: 'color' or 'texture' (LBP).")
    parser.add_argument('--grid_size', type=int, default=4, help='Grid dimensions (N for an N x N grid).')
    parser.add_argument('--color_bins', type=int, default=12, help='Number of bins per channel for COLOR histograms.')
    parser.add_argument('--lbp_points', type=int, default=8,
                        help='Number of points for LBP texture features (angular quantization).')
    parser.add_argument('--lbp_radius', type=int, default=1, help='Radius for LBP texture features.')
    parser.add_argument('--visualize_n', type=int, default=10,
                        help='Generate and save visualization images for the first N processed images.')
    args = parser.parse_args()
    images_path = os.path.join(args.dataset_dir, 'Images')
    if not os.path.isdir(images_path):
        print(f"ERROR: Images directory not found at '{images_path}'. Please check your --dataset_dir path.")
        return
    if args.feature_type == 'color':
        descriptor_subfolder_name = f'spatial_{args.feature_type}_{args.grid_size}x{args.grid_size}_{args.color_bins}bins'
    else:
        descriptor_subfolder_name = f'spatial_{args.feature_type}_{args.grid_size}x{args.grid_size}_{args.lbp_points}P_{args.lbp_radius}R'
    descriptor_output_path = os.path.join(args.output_dir, 'descriptors', descriptor_subfolder_name)
    vis_output_path = os.path.join(args.output_dir, 'visualizations', descriptor_subfolder_name)
    print(f"--- Configuration ---")
    print(f"Dataset Path: {args.dataset_dir}")
    print(f"Output Path: {args.output_dir}")
    print(f"Feature Type: {args.feature_type}")
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    if args.feature_type == 'color':
        print(f"Color Bins: {args.color_bins}")
    else:
        print(f"LBP Points: {args.lbp_points} (Angular Samples)")
        print(f"LBP Radius: {args.lbp_radius}")
    print(f"Descriptor save location: {descriptor_output_path}")
    print(f"Visualization save location: {vis_output_path}")
    print("--------------------")
    os.makedirs(descriptor_output_path, exist_ok=True)
    if args.visualize_n > 0:
        os.makedirs(vis_output_path, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.bmp')])
    for i, filename in enumerate(tqdm(image_files, desc="Processing Images")):
        img_path = os.path.join(images_path, filename)
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"\nDEBUG: Failed to read image: {img_path}. Skipping.")
                continue
        except Exception as e:
            print(f"\nDEBUG: Error reading {img_path}: {e}. Skipping.")
            continue
        h, w, _ = image.shape
        cell_h, cell_w = h // args.grid_size, w // args.grid_size
        all_cell_descriptors = []
        image_with_grid = image.copy()
        for row in range(args.grid_size):
            for col in range(args.grid_size):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = image[y1:y2, x1:x2]
                cv2.rectangle(image_with_grid, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if cell.shape[0] == 0 or cell.shape[1] == 0:
                    print(f"\nDEBUG: Empty cell at ({row},{col}) for {filename}. Appending zeros.")
                    if args.feature_type == 'color':
                        feature_len = args.color_bins ** 3
                    else:
                        feature_len = args.lbp_points + 2
                    cell_descriptor = np.zeros(feature_len)
                else:
                    cell_descriptor = create_feature_descriptor(cell, args.feature_type, args.color_bins,
                                                                args.lbp_points, args.lbp_radius)
                all_cell_descriptors.append(cell_descriptor)
        final_descriptor = np.concatenate(all_cell_descriptors)
        sio.savemat(os.path.join(descriptor_output_path, filename.replace('.bmp', '.mat')), {'F': [final_descriptor]})
        if i < args.visualize_n:
            h_vis, w_vis = image.shape[:2]
            if args.feature_type == 'texture':
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                lbp = local_binary_pattern(gray, P=args.lbp_points, R=args.lbp_radius, method='uniform')
                lbp_vis = np.uint8(255.0 * (lbp / lbp.max()))
                lbp_vis_color = cv2.cvtColor(lbp_vis, cv2.COLOR_GRAY2BGR)
                canvas = np.zeros((h_vis, w_vis * 3, 3), dtype=np.uint8)
                canvas[:, :w_vis] = image
                canvas[:, w_vis:w_vis * 2] = image_with_grid
                canvas[:, w_vis * 2:] = lbp_vis_color
                cv2.putText(canvas, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(canvas, "Gridded Image", (w_vis + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(canvas, f"LBP Texture Map ({args.lbp_points} points)", (w_vis * 2 + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                canvas = np.zeros((h_vis, w_vis * 2, 3), dtype=np.uint8)
                canvas[:, :w_vis] = image
                canvas[:, w_vis:] = image_with_grid
                cv2.putText(canvas, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(canvas, "Gridded Image", (w_vis + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            vis_filename = filename.replace('.bmp', '.jpg')
            cv2.imwrite(os.path.join(vis_output_path, vis_filename), canvas)
    print("\n--- Processing complete! ---")


if __name__ == '__main__':
    main()