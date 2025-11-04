#!/usr/bin/env python3
#
# SCRIPT: compute_hog_features.py
#
# PURPOSE: Computes Histogram of Oriented Gradients (HOG) features for the dataset.
#          HOG is a powerful descriptor for capturing object shape and structure.
#
# EXAMPLE COMMAND:
# python3 compute_hog_features.py --output_dir ./hog_outputs --visualize_n 10
#

import os
import argparse
import shutil
import cv2
import numpy as np
import scipy.io as sio
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm


def create_hog_descriptor(image, orientations, pixels_per_cell, cells_per_block, visualize=False):
    """Computes a HOG descriptor for a single image."""
    # HOG works on grayscale images
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The hog function returns the feature vector and optionally a visualization image
    if visualize:
        features, hog_image = hog(gray_image, orientations=orientations,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  visualize=True, feature_vector=True,
                                  block_norm='L2-Hys')
        return features, hog_image
    else:
        features = hog(gray_image, orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       visualize=False, feature_vector=True,
                       block_norm='L2-Hys')
        return features, None


def main():
    parser = argparse.ArgumentParser(description='Compute HOG features for the MSRC dataset.')
    parser.add_argument('--dataset_dir', type=str, default='./MSRC_ObjCategImageDatabase_v2',
                        help='Path to the root of the MSRC dataset.')
    parser.add_argument('--output_dir', type=str, default='./hog_outputs',
                        help='Path where all outputs will be saved.')
    # HOG-specific parameters for experimentation
    parser.add_argument('--orientations', type=int, default=9, help='Number of orientation bins for HOG.')
    parser.add_argument('--pixels_per_cell', type=int, default=8, help='Size (in pixels) of a HOG cell.')
    parser.add_argument('--cells_per_block', type=int, default=2,
                        help='Number of cells in each direction for a HOG block.')
    parser.add_argument('--visualize_n', type=int, default=10, help='Generate visualization for the first N images.')
    args = parser.parse_args()

    images_path = os.path.join(args.dataset_dir, 'Images')
    if not os.path.isdir(images_path):
        print(f"ERROR: Images directory not found at '{images_path}'.")
        return

    # Create descriptive subfolder names
    descriptor_subfolder_name = f'hog_{args.orientations}o_{args.pixels_per_cell}ppc_{args.cells_per_block}cpb'
    descriptor_output_path = os.path.join(args.output_dir, 'descriptors', descriptor_subfolder_name)
    vis_output_path = os.path.join(args.output_dir, 'visualizations', descriptor_subfolder_name)

    print("--- HOG Feature Computation ---")
    print(
        f"Parameters: {args.orientations} orientations, {args.pixels_per_cell}x{args.pixels_per_cell} cell size, {args.cells_per_block}x{args.cells_per_block} block size")
    print(f"Descriptors will be saved to: {descriptor_output_path}")

    os.makedirs(descriptor_output_path, exist_ok=True)
    if args.visualize_n > 0:
        os.makedirs(vis_output_path, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.bmp')])

    for i, filename in enumerate(tqdm(image_files, desc="Processing Images")):
        img_path = os.path.join(images_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"\nDEBUG: Failed to read image: {img_path}. Skipping.")
            continue

        # For HOG, images are often resized to a standard size for consistent descriptor length
        # Let's use a modest standard size.
        std_size = (128, 128)
        image_resized = cv2.resize(image, std_size)

        features, hog_image = create_hog_descriptor(
            image_resized,
            args.orientations,
            (args.pixels_per_cell, args.pixels_per_cell),
            (args.cells_per_block, args.cells_per_block),
            visualize=(i < args.visualize_n)
        )

        # Save descriptor
        sio.savemat(os.path.join(descriptor_output_path, filename.replace('.bmp', '.mat')), {'F': [features]})

        # Save visualization if generated
        if hog_image is not None:
            # Rescale HOG visualization for better viewing
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hog_image_rescaled = (hog_image_rescaled * 255).astype(np.uint8)
            hog_vis_color = cv2.cvtColor(hog_image_rescaled, cv2.COLOR_GRAY2BGR)

            # Create a side-by-side comparison
            h, w, _ = image_resized.shape
            canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
            canvas[:, :w] = image_resized
            canvas[:, w:] = hog_vis_color
            cv2.putText(canvas, "Original (Resized)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(canvas, "HOG Visualization", (w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            vis_filename = filename.replace('.bmp', '.jpg')
            cv2.imwrite(os.path.join(vis_output_path, vis_filename), canvas)

    print("\n--- HOG processing complete! ---")


if __name__ == '__main__':
    main()