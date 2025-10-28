# In file: optimize_bins.py
import os
import numpy as np
import cv2
import scipy.io as sio
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
BINS_TO_TEST = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
NUM_EVAL_QUERIES = 50
TOP_K_FOR_METRICS = 10
DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
DESCRIPTOR_BASE_FOLDER = 'descriptors'


# --- Helper Functions (unchanged) ---
def get_class_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        return int(basename.split('_')[0])
    except (ValueError, IndexError):
        return -1


def extract_histogram(img, n_bins):
    img_8bit = (img * 255).astype(np.uint8)
    hist = cv2.calcHist([img_8bit], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten()


def cvpr_compare(f1, f2):
    return np.linalg.norm(f1 - f2)


# --- Main Optimization Script ---
print("Starting histogram bin optimization process...")
print(f"Testing bin counts: {BINS_TO_TEST}")
print(f"Using {NUM_EVAL_QUERIES} evaluation queries per bin count.")

results_avg_precision = []
results_avg_dist = []
image_files = [f for f in os.listdir(os.path.join(DATASET_FOLDER, 'Images')) if f.endswith(".bmp")]
total_files = len(image_files)

# --- The Main Loop ---
for n_bins in BINS_TO_TEST:
    print(f"\n===== Processing for {n_bins} bins per channel =====")

    # Step 1: Compute descriptors
    out_subfolder = f'temp_global_rgb_{n_bins}bins'
    out_dir = os.path.join(DESCRIPTOR_BASE_FOLDER, out_subfolder)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    print(f"Generating {total_files} descriptors...")
    for filename in tqdm(image_files, desc=f"Computing ({n_bins} bins)"):
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
        img = cv2.imread(img_path)
        img_normalized = img.astype(np.float64) / 255.0
        F = extract_histogram(img_normalized, n_bins)
        sio.savemat(os.path.join(out_dir, filename.replace('.bmp', '.mat')), {'F': [F]})

    # Step 2: Load descriptors
    ALLFEAT, ALLFILES = [], []
    for filename in os.listdir(out_dir):
        img_actual_path = os.path.join(DATASET_FOLDER, 'Images', filename.replace(".mat", ".bmp"))
        descriptor_path = os.path.join(out_dir, filename)
        mat_data = sio.loadmat(descriptor_path)
        ALLFILES.append(img_actual_path)
        ALLFEAT.append(mat_data['F'][0])
    ALLFEAT = np.array(ALLFEAT)

    # Step 3: Evaluate performance
    current_bin_precisions = []
    current_bin_distances = []
    print(f"Running {NUM_EVAL_QUERIES} evaluation queries...")
    for _ in tqdm(range(NUM_EVAL_QUERIES), desc=f"Evaluating ({n_bins} bins)"):
        query_idx = np.random.randint(0, total_files)
        query_feat = ALLFEAT[query_idx]
        query_class = get_class_from_filename(ALLFILES[query_idx])

        dst = [(cvpr_compare(query_feat, feat), i) for i, feat in enumerate(ALLFEAT)]
        dst.sort(key=lambda x: x[0])

        top_k_results = dst[1: TOP_K_FOR_METRICS + 1]
        top_k_distances = [d[0] for d in top_k_results]
        current_bin_distances.append(np.mean(top_k_distances))

        correct_matches = sum(
            1 for _, result_idx in top_k_results if get_class_from_filename(ALLFILES[result_idx]) == query_class)
        precision = correct_matches / TOP_K_FOR_METRICS
        current_bin_precisions.append(precision)

    # Step 4: Average and store metrics
    avg_precision = np.mean(current_bin_precisions)
    avg_distance = np.mean(current_bin_distances)
    results_avg_precision.append(avg_precision)
    results_avg_dist.append(avg_distance)
    print(f"Results for {n_bins} bins: Avg Precision = {avg_precision:.4f}, Avg Distance = {avg_distance:.4f}")

    # #######################################################################
    # #######                    NEW CLEANUP STEP                     #######
    # #######################################################################
    # After we have the results for this bin count, delete its temp folder.
    print(f"Cleaning up temporary directory: {out_dir}")
    shutil.rmtree(out_dir)  # <-- THIS IS THE NEW LINE
    # #######################################################################

# Step 5: Plotting the final results (unchanged)
print("\n--- All tests complete. Generating final plot... ---")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Visual Search Performance vs. Histogram Bins', fontsize=16)

# Plot 1: Average Precision @ 10
ax1.plot(BINS_TO_TEST, results_avg_precision, marker='o', linestyle='-', color='b')
ax1.set_title('Search Quality (Higher is Better)', fontsize=12)
ax1.set_xlabel('Number of Bins per Channel')
ax1.set_ylabel(f'Average Precision@{TOP_K_FOR_METRICS}')
ax1.grid(True, linestyle='--')
best_precision_idx = np.argmax(results_avg_precision)
best_bins_p = BINS_TO_TEST[best_precision_idx]
best_precision = results_avg_precision[best_precision_idx]
ax1.axvline(x=best_bins_p, color='g', linestyle='--', label=f'Peak Precision at {best_bins_p} bins')
ax1.annotate(f'Best: {best_precision:.3f}', xy=(best_bins_p, best_precision),
             xytext=(best_bins_p + 5, best_precision - 0.01), arrowprops=dict(facecolor='green', shrink=0.05))
ax1.legend()

# Plot 2: Average Distance of Top Matches
ax2.plot(BINS_TO_TEST, results_avg_dist, marker='s', linestyle='-', color='r')
ax2.set_title('Feature Space Compactness', fontsize=12)
ax2.set_xlabel('Number of Bins per Channel')
ax2.set_ylabel(f'Average Distance of Top {TOP_K_FOR_METRICS} Matches')
ax2.grid(True, linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_image_path = 'bin_optimization_results.png'
plt.savefig(output_image_path)
print(f"Success! Output plot saved to '{output_image_path}'")
plt.show()