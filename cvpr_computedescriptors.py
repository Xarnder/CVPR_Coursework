# In file: cvpr_computedescriptors.py
import os
import numpy as np
import cv2
import scipy.io as sio
import shutil  # For deleting directories
import argparse  # For command-line arguments
from extract_histogram import extract_histogram

# --- Configuration ---
DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
OUT_FOLDER = 'descriptors'
OUT_SUBFOLDER = 'global_rgb_2bins'

# --- Main Script Logic ---

# 1. Setup argument parser
parser = argparse.ArgumentParser(description='Compute image descriptors.')
parser.add_argument('--clean', action='store_true',
                    help='Remove the descriptor directory before running.')
args = parser.parse_args()

# 2. Define output directory and handle the --clean flag
out_dir = os.path.join(OUT_FOLDER, OUT_SUBFOLDER)

if args.clean:
    if os.path.exists(out_dir):
        print(f"--- '--clean' flag used. Deleting old descriptor directory: {out_dir} ---")
        shutil.rmtree(out_dir)
    else:
        print(f"--- '--clean' flag used. Directory not found, no action needed. ---")

# 3. Ensure the output directory exists (it might have just been deleted)
os.makedirs(out_dir, exist_ok=True)

# 4. Get list of files and start processing loop
all_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith(".bmp")]
total_files = len(all_files)
print(f"Found {total_files} images. Starting descriptor computation...")

for idx, filename in enumerate(all_files):
    print(f'\rProcessing image {idx + 1}/{total_files}: {filename}', end='', flush=True)

    img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"\nWarning: Could not read image {img_path}. Skipping.")
        continue

    img_normalized = img.astype(np.float64) / 255.0
    fout = os.path.join(out_dir, filename.replace('.bmp', '.mat'))

    F = extract_histogram(img_normalized)

    sio.savemat(fout, {'F': [F]})

print("\n--- Descriptor computation complete! ---")