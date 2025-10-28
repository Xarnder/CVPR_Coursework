# In file: evaluate_system.py
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay

# --- Configuration ---
# Set this to the number of bins for the experiment you want to evaluate.
N_BINS_PER_CHANNEL = 12

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = f'global_rgb_{N_BINS_PER_CHANNEL}bins'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

# For the confusion matrix, we'll make a prediction based on the majority
# vote of the top K results. 5 is a common choice.
K_FOR_PREDICTION = 5


# --- Helper Functions ---

def get_class_from_filename(filename):
    """Extracts the class number from a filename (e.g., '7_19_s.bmp' -> 7)."""
    basename = os.path.basename(filename)
    try:
        return int(basename.split('_')[0])
    except (ValueError, IndexError):
        return -1  # Return an invalid class if filename is not as expected


def cvpr_compare(f1, f2):
    """Calculates Euclidean distance between two feature vectors."""
    return np.linalg.norm(f1 - f2)


# --- Main Evaluation Script ---

print(f"--- Starting Full System Evaluation for {N_BINS_PER_CHANNEL}-bin Histograms ---")

# 1. Load All Descriptors and Ground Truth Class Labels
print(f"Loading descriptors from '{DESCRIPTOR_SUBFOLDER}'...")
descriptor_dir = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)
if not os.path.exists(descriptor_dir):
    print(f"\nERROR: Descriptors not found. Please run cvpr_computedescriptors.py for {N_BINS_PER_CHANNEL} bins first.")
    exit()

ALLFEAT, ALLFILES = [], []
for filename in os.listdir(descriptor_dir):
    if filename.endswith('.mat'):
        img_path = os.path.join(IMAGE_FOLDER, 'Images', filename.replace(".mat", ".bmp"))
        descriptor_path = os.path.join(descriptor_dir, filename)
        mat_data = sio.loadmat(descriptor_path)
        ALLFILES.append(img_path)
        ALLFEAT.append(mat_data['F'][0])

ALLFEAT = np.array(ALLFEAT)
NIMG = len(ALLFILES)
ALLCLASSES = np.array([get_class_from_filename(f) for f in ALLFILES])
print(f"Loaded {NIMG} descriptors.")

# Prepare lists to store results from all queries
y_true_for_cm = []  # Ground truth labels for the confusion matrix
y_pred_for_cm = []  # Predicted labels for the confusion matrix
all_precisions_for_pr = []  # A list to hold the precision vectors for each query

# 2. Main Loop: Iterate through EVERY image in the dataset as a query
for query_idx in tqdm(range(NIMG), desc="Evaluating all 591 queries"):
    query_feat = ALLFEAT[query_idx]
    query_class = ALLCLASSES[query_idx]

    # Calculate how many other images of the same class exist in the dataset
    total_relevant_items = np.sum(ALLCLASSES == query_class) - 1

    # Perform the search: calculate distance to all other images
    dst = [(cvpr_compare(query_feat, feat), i) for i, feat in enumerate(ALLFEAT)]
    dst.sort(key=lambda x: x[0])

    # Get the ranked list of indices and classes (excluding the query itself)
    ranked_indices = [d[1] for d in dst[1:]]
    ranked_classes = ALLCLASSES[ranked_indices]

    # --- A) Calculate Precision-Recall values for this specific query ---
    relevant_found_count = 0
    precisions = []
    recalls = []
    for k, result_class in enumerate(ranked_classes):
        if result_class == query_class:
            relevant_found_count += 1

        precision = relevant_found_count / (k + 1)
        recall = relevant_found_count / total_relevant_items if total_relevant_items > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    # To average PR curves, we interpolate precision at standard recall levels
    recall_levels = np.linspace(0.1, 1.0, 10)
    interpolated_precisions = []
    for r_level in recall_levels:
        # Find max precision for any recall value >= current level
        p_max = max([p for r, p in zip(recalls, precisions) if r >= r_level], default=0)
        interpolated_precisions.append(p_max)
    all_precisions_for_pr.append(interpolated_precisions)

    # --- B) Determine the predicted class for the Confusion Matrix ---
    top_k_classes = ranked_classes[:K_FOR_PREDICTION]
    # Predict the class by majority vote among the top K results
    predicted_class = Counter(top_k_classes).most_common(1)[0][0]

    y_true_for_cm.append(query_class)
    y_pred_for_cm.append(predicted_class)

# 3. Final Aggregation and Plotting
print("\n--- Evaluation complete. Generating final plots... ---")

# --- Plot 1: The Precision-Recall Curve ---
mean_precisions = np.mean(all_precisions_for_pr, axis=0)
mAP = np.mean(mean_precisions)  # mean Average Precision (a single score for system quality)

plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0.1, 1.0, 10), mean_precisions, marker='o', linestyle='-', color='b')
plt.title(f'Mean Precision-Recall Curve ({N_BINS_PER_CHANNEL}-bin Histogram)\nmAP = {mAP:.4f}', fontsize=16)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Mean Precision', fontsize=12)
plt.grid(True, linestyle='--')
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.tight_layout()
pr_curve_path = f'pr_curve_{N_BINS_PER_CHANNEL}bins.png'
plt.savefig(pr_curve_path)
print(f"Precision-Recall curve saved to '{pr_curve_path}'")
plt.show()

# --- Plot 2: The Confusion Matrix ---
fig, ax = plt.subplots(figsize=(12, 12))
# The class labels for the plot axes
class_labels = [str(i) for i in sorted(np.unique(ALLCLASSES))]

ConfusionMatrixDisplay.from_predictions(y_true_for_cm, y_pred_for_cm,
                                        labels=sorted(np.unique(ALLCLASSES)),
                                        display_labels=class_labels,
                                        ax=ax, cmap='Blues', colorbar=True)

plt.title(f'Confusion Matrix ({N_BINS_PER_CHANNEL}-bin Histogram)', fontsize=16)
ax.set_xlabel('Predicted Class', fontsize=12)
ax.set_ylabel('True Class', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
confusion_matrix_path = f'confusion_matrix_{N_BINS_PER_CHANNEL}bins.png'
plt.savefig(confusion_matrix_path)
print(f"Confusion matrix saved to '{confusion_matrix_path}'")
plt.show()