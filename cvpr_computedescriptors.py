# In file: cvpr_computedescriptors.py
import os
import numpy as np
import cv2
import scipy.io as sio
import shutil
import argparse
from tqdm import tqdm
from extract_histogram import extract_histogram

# --- Main Script Logic ---

def main():
    # 1. Setup argument parser for command-line control
    parser = argparse.ArgumentParser(description='Compute global color histogram descriptors.')
    parser.add_argument('--bins', type=int, default=12,
                        help='Number of quantization bins per RGB channel (the value Q).')
    parser.add_argument('--clean', action='store_true',
                        help='Remove the descriptor directory before running.')
    args = parser.parse_args()

    # --- Configuration ---
    Q = args.bins
    DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
    OUT_BASE_FOLDER = 'descriptors'
    # Create a descriptive subfolder name based on the number of bins
    OUT_SUBFOLDER = f'global_rgb_{Q}bins'

    # 2. Define output directory and handle the --clean flag
    out_dir = os.path.join(OUT_BASE_FOLDER, OUT_SUBFOLDER)

    if args.clean:
        if os.path.exists(out_dir):
            print(f"--- '--clean' flag used. Deleting old descriptor directory: {out_dir} ---")
            shutil.rmtree(out_dir)
        else:
            print(f"--- '--clean' flag used. Directory not found, no action needed. ---")

    # 3. Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    print(f"--- Starting Descriptor Computation ---")
    print(f"Quantization level (Q): {Q} bins per channel")
    print(f"Output will be saved to: {out_dir}")

    # 4. Get list of image files
    image_dir = os.path.join(DATASET_FOLDER, 'Images')
    all_files = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
    total_files = len(all_files)
    print(f"Found {total_files} images to process.")

    # 5. Start processing loop with a progress bar
    for filename in tqdm(all_files, desc="Computing Descriptors"):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"\nWarning: Could not read image {img_path}. Skipping.")
            continue

        # Normalize image to the range [0.0, 1.0] as required by our function
        img_normalized = img.astype(np.float64) / 255.0

        # Call the correct histogram function with the specified number of bins
        F = extract_histogram(img_normalized, q_val=Q)

        # Save the descriptor to a .mat file
        fout = os.path.join(out_dir, filename.replace('.bmp', '.mat'))
        sio.savemat(fout, {'F': [F]})

    print(f"\n--- Descriptor computation complete! ---")
    print(f"All {total_files} descriptors are saved in '{out_dir}'.")


if __name__ == '__main__':
    main()