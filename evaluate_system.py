# evaluate_system.py
"""
Requirement 2 evaluator (final version):
- Automatically discovers ALL descriptor subfolders in the './descriptors' directory.
- Computes PR statistics (including mAP, Precision@K, and Recall@K) for each descriptor set.
- Plots PR curves for ALL experiments on a single, combined figure with a clean legend.
- Computes and saves a confusion matrix per experiment (k-NN over neighbours).
- Saves a master CSV summary with key metrics, ranked by performance.
- Fixes the memory leak that caused slowdowns during sequential evaluations.

Usage example:
  python evaluate_system.py --topk 10 --knn 5 --with-prk-points
"""

import argparse
import os
import re
import csv
from collections import Counter

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay

# ---- Configuration ----
DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
DESCRIPTOR_FOLDER = 'descriptors'
TOPK_DEFAULT = 10
KNN_DEFAULT = 5


# ---- Helper Functions ----
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_class_from_filename(path_or_name: str) -> str:
    name = os.path.basename(path_or_name)
    stem = os.path.splitext(name)[0]
    return stem.split('_', 1)[0]


def load_experiment(descriptor_subdir: str):
    desc_dir = os.path.join(DESCRIPTOR_FOLDER, descriptor_subdir)
    if not os.path.isdir(desc_dir):
        raise FileNotFoundError(f"Descriptor directory not found: {desc_dir}")

    feats, files = [], []
    for filename in sorted(os.listdir(desc_dir)):
        if not filename.endswith('.mat'):
            continue
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename.replace('.mat', '.bmp'))
        if not os.path.exists(img_path): continue
        mat = sio.loadmat(os.path.join(desc_dir, filename))
        if 'F' not in mat:
            raise KeyError(f"'F' not found in {filename}")
        F = mat['F'].ravel().astype(np.float64)
        feats.append(F);
        files.append(img_path)

    if not files:
        raise RuntimeError(f"No .mat descriptors found in {desc_dir}")

    return np.vstack(feats), list(files), np.array([get_class_from_filename(f) for f in files])


def rank_for_query(query_idx: int, FEAT: np.ndarray):
    q = FEAT[query_idx]
    d = np.linalg.norm(FEAT - q, axis=1)
    order = np.argsort(d)
    return order[order != query_idx]


def compute_pr_curve_per_query(query_idx: int, FEAT: np.ndarray, CLASSES: np.ndarray):
    true_class = CLASSES[query_idx]
    ranked = rank_for_query(query_idx, FEAT)
    ranked_classes = CLASSES[ranked]

    total_relevant = int(np.sum(CLASSES == true_class)) - 1
    if total_relevant <= 0:
        return np.array([1.0]), np.array([0.0]), ranked

    precisions, recalls = [], []
    tp = 0
    for i, c in enumerate(ranked_classes, start=1):
        if c == true_class:
            tp += 1
        precisions.append(tp / i)
        recalls.append(tp / total_relevant)

    return np.array(precisions, float), np.array(recalls, float), ranked


def interpolate_pr(precisions: np.ndarray, recalls: np.ndarray, recall_samples=None):
    if recall_samples is None:
        recall_samples = np.linspace(0.0, 1.0, 11)
    interp = np.zeros_like(recall_samples, dtype=float)
    for i, r in enumerate(recall_samples):
        mask = recalls >= r
        interp[i] = np.max(precisions[mask]) if np.any(mask) else 0.0
    return recall_samples, interp


def compute_confusion_matrix_preds(FEAT: np.ndarray, CLASSES: np.ndarray, knn: int):
    y_true, y_pred = [], []
    for i in range(len(CLASSES)):
        ranked = rank_for_query(i, FEAT)
        neigh = ranked[:knn]
        vote = Counter(CLASSES[neigh]).most_common(1)[0][0]
        y_true.append(CLASSES[i]);
        y_pred.append(vote)
    return np.array(y_true), np.array(y_pred)


def discover_experiment_folders():
    """Discovers ALL subdirectories within the main descriptor folder."""
    if not os.path.isdir(DESCRIPTOR_FOLDER):
        return []
    subfolders = [name for name in os.listdir(DESCRIPTOR_FOLDER)
                  if os.path.isdir(os.path.join(DESCRIPTOR_FOLDER, name))]
    subfolders.sort()
    return subfolders


def generate_clean_label(subfolder: str, mAP: float) -> str:
    """Creates a cleaner, more readable label for the plot legend."""
    label = subfolder
    if 'global_rgb' in subfolder:
        m = re.search(r'(\d+)bins', subfolder)
        if m: label = f"Global Colour ({m.group(1)} bins)"
    elif 'spatial_color' in subfolder:
        m = re.search(r'(\d+x\d+)_(\d+)bins', subfolder)
        if m: label = f"Spatial Colour ({m.group(1)}, {m.group(2)}b)"
    elif 'spatial_texture' in subfolder:
        m = re.search(r'(\d+x\d+)_(\d+P)', subfolder)
        if m: label = f"Spatial LBP ({m.group(1)}, {m.group(2)})"
    elif 'hog' in subfolder:
        label = "HOG"
    return f"{label} (mAP={mAP:.3f})"


def evaluate_experiment(subfolder: str, topk: int, knn: int):
    print(f"\n=== Evaluating experiment: {subfolder} ===")
    FEAT, FILES, CLASSES = load_experiment(subfolder)
    N = len(FILES)
    classes_sorted = sorted(np.unique(CLASSES).tolist())

    recall_grid = np.linspace(0.0, 1.0, 11)
    interp_precisions, P_at_K, R_at_K = [], [], []

    for qi in tqdm(range(N), desc=f"PR per query [{subfolder}]"):
        P, R, ranked = compute_pr_curve_per_query(qi, FEAT, CLASSES)
        _, Pint = interpolate_pr(P, R, recall_grid)
        interp_precisions.append(Pint)

        true_class = CLASSES[qi]
        ranked_classes = CLASSES[ranked]
        tp_at_k = int(np.sum(ranked_classes[:topk] == true_class))
        total_relevant = int(np.sum(CLASSES == true_class)) - 1
        prec_at_k = tp_at_k / topk
        rec_at_k = (tp_at_k / total_relevant) if total_relevant > 0 else 0.0
        P_at_K.append(prec_at_k);
        R_at_K.append(rec_at_k)

    mean_interp_precision = np.mean(np.stack(interp_precisions, axis=0), axis=0)
    mAP11 = float(np.mean(mean_interp_precision))
    mean_P_at_K = float(np.mean(P_at_K))
    mean_R_at_K = float(np.mean(R_at_K))

    # Confusion matrix
    y_true, y_pred = compute_confusion_matrix_preds(FEAT, CLASSES, knn=knn)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=classes_sorted, display_labels=classes_sorted,
        ax=ax, cmap='Blues', colorbar=True, xticks_rotation='vertical'
    )
    ax.set_title(f'Confusion Matrix — {subfolder} (k={knn})')
    ax.set_xlabel('Predicted');
    ax.set_ylabel('True')
    plt.tight_layout()
    ensure_dir('plots')
    cm_path = os.path.join('plots', f'confusion_matrix_{subfolder}.png')
    plt.savefig(cm_path, dpi=150)

    # ######################################################################
    # #######                  THE CRITICAL FIX IS HERE                #######
    # ####### This line prevents the Matplotlib memory leak and slowdown.#######
    # ######################################################################
    plt.close(fig)

    print(f"Saved confusion matrix: {cm_path}")

    metrics = {
        'experiment': subfolder, 'mAP11': mAP11, f'Precision@{topk}': mean_P_at_K,
        f'Recall@{topk}': mean_R_at_K, 'num_images': N, 'knn_for_cm': knn
    }
    pr_payload = {
        'recall_grid': recall_grid, 'mean_precision_curve': mean_interp_precision,
        'label': generate_clean_label(subfolder, mAP11), 'mAP': mAP11,
        'mean_P_at_K': mean_P_at_K, 'mean_R_at_K': mean_R_at_K, 'name': subfolder
    }
    return metrics, pr_payload


def main():
    parser = argparse.ArgumentParser(
        description="Requirement 2 evaluator: PR curves (combined), PR@K, confusion matrices.")
    parser.add_argument('--topk', type=int, default=TOPK_DEFAULT,
                        help=f"K for Precision@K and Recall@K (default {TOPK_DEFAULT}).")
    parser.add_argument('--knn', type=int, default=KNN_DEFAULT,
                        help=f"k for confusion-matrix voting (default {KNN_DEFAULT}).")
    parser.add_argument('--with-prk-points', action='store_true',
                        help="Overlay mean (Recall@K, Precision@K) point per experiment on the combined plot.")
    args = parser.parse_args()

    exp_subfolders = discover_experiment_folders()
    if not exp_subfolders:
        raise RuntimeError(f"No experiment folders found under '{DESCRIPTOR_FOLDER}'.")

    print("Discovered experiments to evaluate:")
    for s in exp_subfolders:
        print(" -", s)

    all_metrics, curve_payloads = [], []
    for sub in exp_subfolders:
        metrics, curve = evaluate_experiment(subfolder=sub, topk=args.topk, knn=args.knn)
        all_metrics.append(metrics);
        curve_payloads.append(curve)

    # Save CSV summary
    ensure_dir('plots')
    csv_path = os.path.join('plots', 'master_performance_summary.csv')
    fieldnames = ['experiment', 'mAP11', f'Precision@{args.topk}', f'Recall@{args.topk}', 'num_images', 'knn_for_cm']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_metrics, key=lambda r: r['mAP11'], reverse=True):
            writer.writerow(row)
    print(f"\nSaved master performance summary: {csv_path}")

    # --- Combined PR plot: ALL experiments on ONE graph ---
    fig, ax = plt.subplots(figsize=(12, 8))
    for payload in sorted(curve_payloads, key=lambda p: p['mAP'], reverse=True):
        ax.plot(payload['recall_grid'], payload['mean_precision_curve'], marker='o', markersize=4,
                label=payload['label'])

    if args.with_prk_points:
        for payload in curve_payloads:
            ax.scatter(payload['mean_R_at_K'], payload['mean_P_at_K'], s=60, marker='X', zorder=5)
            # Annotation for the point
            ax.annotate(f"P@{args.topk}", (payload['mean_R_at_K'], payload['mean_P_at_K']),
                        textcoords="offset points", xytext=(5, -10), fontsize=8, color='darkred')

    ax.set_xlim(0, 1.01);
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Recall');
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves for All Descriptors')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    combined_path = os.path.join('plots', 'pr_curves_all_experiments.png')
    plt.savefig(combined_path, dpi=150)
    print(f"Saved combined PR curves: {combined_path}")
    plt.show()


if __name__ == '__main__':
    main()