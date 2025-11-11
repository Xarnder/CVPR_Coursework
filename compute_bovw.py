#!/usr/bin/env python3
#
# SCRIPT: compute_bovw.py
#
# PURPOSE: Implements the Bag of Visual Words (BoVW) pipeline for Requirement 6.
#          1. Extracts SIFT features from all images in the dataset.
#          2. Learns a "visual vocabulary" (codebook) by clustering these features using MiniBatchKMeans.
#          3. For each image, creates a histogram of its visual words to form the final BoVW descriptor.
#
# EXAMPLE COMMAND:
# # This will create a vocabulary of 500 words and save the descriptors
# python3 compute_bovw.py --k 500
#
# # To force re-computation of the codebook even if it exists
# python3 compute_bovw.py --k 500 --recompute_codebook
#

import os
import argparse
import cv2
import numpy as np
import scipy.io as sio
from sklearn.cluster import MiniBatchKMeans
import pickle
from tqdm import tqdm

# --- Configuration ---
DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'


def extract_local_sift_features(image_path: str, sift_detector):
    """Detects keypoints and computes SIFT descriptors for a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # SIFT detects keypoints and computes their 128-dimensional descriptors
    keypoints, descriptors = sift_detector.detectAndCompute(img, None)
    return descriptors


def build_codebook(all_descriptors: np.ndarray, k: int):
    """
    Builds the visual vocabulary (codebook) by clustering all local descriptors
    using MiniBatchKMeans.
    """
    print(f"Clustering {len(all_descriptors)} features into {k} visual words using MiniBatchKMeans...")

    # Using MiniBatchKMeans for speed and memory efficiency
    kmeans = MiniBatchKMeans(n_clusters=k,
                             random_state=42,
                             batch_size=256,
                             n_init='auto',
                             max_iter=100)
    kmeans.fit(all_descriptors)
    return kmeans


def create_bovw_descriptor(local_descriptors: np.ndarray, kmeans_model):
    """
    Creates a BoVW histogram for a single image's local descriptors.
    """
    k = kmeans_model.n_clusters

    # If no descriptors were found in the image, return a zero-vector
    if local_descriptors is None or len(local_descriptors) == 0:
        return np.zeros(k, dtype=np.float32)

    # For each local descriptor, find the closest visual word in the codebook
    visual_words = kmeans_model.predict(local_descriptors)

    # Create a histogram of the visual words
    hist, _ = np.histogram(visual_words, bins=np.arange(k + 1))

    # Normalize the histogram to be a probability distribution (L1 norm)
    hist = hist.astype(np.float32)
    if np.sum(hist) > 0:
        hist /= np.sum(hist)

    return hist


def main():
    parser = argparse.ArgumentParser(description="Generate Bag of Visual Words descriptors.")
    parser.add_argument('--k', type=int, required=True,
                        help="The number of visual words (vocabulary size) for the codebook.")
    parser.add_argument('--output_base_dir', type=str, default='./bovw_outputs',
                        help="Base directory to save the codebook and final descriptors.")
    parser.add_argument('--recompute_codebook', action='store_true',
                        help="Force re-computation of the k-Means codebook even if a saved one exists.")
    args = parser.parse_args()

    images_dir = os.path.join(DATASET_FOLDER, 'Images')
    if not os.path.isdir(images_dir):
        print(f"FATAL: Dataset image directory not found at '{images_dir}'.")
        return

    # --- 1. Setup Paths ---
    # Create descriptive subfolder names for outputs
    descriptor_subfolder_name = f'bovw_{args.k}k'
    descriptor_output_path = os.path.join(args.output_base_dir, 'descriptors', descriptor_subfolder_name)
    codebook_path = os.path.join(args.output_base_dir, 'codebooks')
    codebook_file = os.path.join(codebook_path, f'codebook_{args.k}k.pkl')

    os.makedirs(descriptor_output_path, exist_ok=True)
    os.makedirs(codebook_path, exist_ok=True)

    print("--- Starting Bag of Visual Words Pipeline ---")
    print(f"Vocabulary Size (k): {args.k}")
    print(f"Descriptors will be saved to: {descriptor_output_path}")

    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.bmp')])
    sift = cv2.SIFT_create()

    # --- 2. Build or Load the Codebook ---
    kmeans_model = None
    if os.path.exists(codebook_file) and not args.recompute_codebook:
        print(f"Loading existing codebook from '{codebook_file}'...")
        with open(codebook_file, 'rb') as f:
            kmeans_model = pickle.load(f)
    else:
        print("No existing codebook found or re-computation forced. Building a new one.")

        # Step 2a: Extract local SIFT features from ALL images
        all_sift_descriptors = []
        print("Extracting SIFT features from all images to build vocabulary...")
        for img_path in tqdm(image_files, desc="Extracting SIFT"):
            descriptors = extract_local_sift_features(img_path, sift)
            if descriptors is not None:
                all_sift_descriptors.append(descriptors)

        if not all_sift_descriptors:
            print("FATAL: No SIFT features could be extracted from any image. Cannot build codebook.")
            return

        all_sift_descriptors = np.vstack(all_sift_descriptors)

        # Step 2b: Cluster features to create the codebook
        kmeans_model = build_codebook(all_sift_descriptors, args.k)

        # Step 2c: Save the trained model for future use
        print(f"Saving new codebook to '{codebook_file}'...")
        with open(codebook_file, 'wb') as f:
            pickle.dump(kmeans_model, f)

    if kmeans_model is None:
        print("FATAL: k-Means model could not be loaded or trained.")
        return

    # --- 3. Generate BoVW Descriptors for Each Image ---
    print("\nGenerating BoVW descriptors for each image using the codebook...")
    for img_path in tqdm(image_files, desc="Generating BoVW"):
        # Extract local features for this specific image
        local_descriptors = extract_local_sift_features(img_path, sift)

        # Create the BoVW histogram descriptor
        bovw_descriptor = create_bovw_descriptor(local_descriptors, kmeans_model)

        # Save the descriptor to a .mat file
        filename_stem = os.path.splitext(os.path.basename(img_path))[0]
        output_mat_path = os.path.join(descriptor_output_path, f"{filename_stem}.mat")
        sio.savemat(output_mat_path, {'F': [bovw_descriptor]})

    print("\n--- BoVW Pipeline Complete! ---")
    print(f"All {len(image_files)} descriptors are saved in '{descriptor_output_path}'.")
    print("You can now copy this folder into './descriptors' to run the evaluation scripts.")


if __name__ == '__main__':
    main()