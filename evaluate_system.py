# evaluate_system.py
"""
Requirement 2 evaluator (updated):
- Computes PR statistics (including Precision@K and Recall@K) for EACH experiment.
- Plots PR curves for ALL experiments on ONE combined figure.
- Optionally marks the PR@K point (mean R@K, mean P@K) per experiment on the combined figure.
- Computes a confusion matrix per experiment (k-NN over neighbours).
- Saves a CSV summary with key metrics.

Usage examples:
  Auto-discover experiments under descriptors/global_rgb_*bins:
    python evaluate_system.py --topk 10 --knn 5 --with-prk-points --save-individual-pr

  Or specify bin counts explicitly:
    python evaluate_system.py --experiments 12 24 32 --topk 10 --knn 5 --with-prk-points
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

# ---- Defaults (override via CLI) ----
DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
DESCRIPTOR_FOLDER = 'descriptors'
EXPERIMENT_FOLDER_FMT = 'global_rgb_{bins}bins'
TOPK_DEFAULT = 10
KNN_DEFAULT = 5

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
    for filename in os.listdir(desc_dir):
        if not filename.endswith('.mat'):
            continue
        img_path = os.path.join(DATASET_FOLDER, 'Images', filename.replace('.mat', '.bmp'))
        mat = sio.loadmat(os.path.join(desc_dir, filename))
        if 'F' not in mat:
            raise KeyError(f"'F' not found in {filename}")
        F = mat['F'].ravel().astype(np.float64)
        feats.append(F); files.append(img_path)

    if not files:
        raise RuntimeError(f"No .mat descriptors found in {desc_dir}")

    ALLFEAT = np.vstack(feats)
    ALLFILES = list(files)
    ALLCLASSES = np.array([get_class_from_filename(f) for f in ALLFILES])
    return ALLFEAT, ALLFILES, ALLCLASSES

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
        return np.array([1.0]), np.array([0.0]), ranked  # degenerate

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
        y_true.append(CLASSES[i]); y_pred.append(vote)
    return np.array(y_true), np.array(y_pred)

def discover_experiment_folders():
    results = []
    if not os.path.isdir(DESCRIPTOR_FOLDER):
        return results
    pat = re.compile(r'^global_rgb_(\d+)bins$')
    for name in os.listdir(DESCRIPTOR_FOLDER):
        m = pat.match(name)
        if m:
            results.append((int(m.group(1)), name))
    results.sort()
    return results

def evaluate_experiment(subfolder: str, topk: int, knn: int, save_individual_pr: bool):
    print(f"\n=== Evaluating experiment: {subfolder} ===")
    FEAT, FILES, CLASSES = load_experiment(subfolder)
    N = len(FILES)
    classes_sorted = sorted(np.unique(CLASSES).tolist())

    recall_grid = np.linspace(0.0, 1.0, 11)
    interp_precisions = []
    P_at_K, R_at_K = [], []

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
        P_at_K.append(prec_at_k); R_at_K.append(rec_at_k)

    mean_interp_precision = np.mean(np.stack(interp_precisions, axis=0), axis=0)
    mAP11 = float(np.mean(mean_interp_precision))
    mean_P_at_K = float(np.mean(P_at_K))
    mean_R_at_K = float(np.mean(R_at_K))

    # Confusion matrix
    y_true, y_pred = compute_confusion_matrix_preds(FEAT, CLASSES, knn=knn)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=classes_sorted,
        display_labels=classes_sorted,
        ax=ax, cmap='Blues', colorbar=True
    )
    ax.set_title(f'Confusion Matrix — {subfolder} (k={knn})')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ensure_dir('plots')
    cm_path = os.path.join('plots', f'confusion_matrix_{subfolder}.png')
    plt.savefig(cm_path, dpi=150); plt.close(fig)
    print(f"Saved confusion matrix: {cm_path}")

    m = re.search(r'(\d+)bins', subfolder)
    bins = int(m.group(1)) if m else None

    metrics = {
        'experiment': subfolder,
        'bins': bins,
        'mAP11': mAP11,
        f'Precision@{topk}': mean_P_at_K,
        f'Recall@{topk}': mean_R_at_K,
        'num_images': N,
        'knn_for_cm': knn
    }

    # Return richer curve payload so the combined plot can also draw PR@K markers.
    pr_payload = {
        'recall_grid': recall_grid,
        'mean_precision_curve': mean_interp_precision,
        'label': f'{bins} bins — mAP11={mAP11:.3f}',
        'bins': bins,
        'mean_P_at_K': mean_P_at_K,
        'mean_R_at_K': mean_R_at_K
    }

    # Optional per-experiment PR plot
    if save_individual_pr:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(recall_grid, mean_interp_precision, marker='o')
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
        ax2.set_title(f'PR Curve — {bins} bins\n'
                      f'mAP11={mAP11:.3f} | P@{topk}={mean_P_at_K:.3f} | R@{topk}={mean_R_at_K:.3f}')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = os.path.join('plots', f'pr_curve_{subfolder}.png')
        plt.savefig(pr_path, dpi=150); plt.close(fig2)
        print(f"Saved PR curve: {pr_path}")

    return metrics, pr_payload

def main():
    parser = argparse.ArgumentParser(description="Requirement 2 evaluator: PR curves (combined), PR@K, confusion matrices.")
    parser.add_argument('--experiments', nargs='*', type=int,
                        help="Bin counts to evaluate (e.g., --experiments 12 24 32). "
                             "If omitted, auto-discovers 'global_rgb_*bins'.")
    parser.add_argument('--topk', type=int, default=TOPK_DEFAULT,
                        help=f"K for Precision@K and Recall@K (default {TOPK_DEFAULT}).")
    parser.add_argument('--knn', type=int, default=KNN_DEFAULT,
                        help=f"k for confusion-matrix voting (default {KNN_DEFAULT}).")
    parser.add_argument('--save-individual-pr', action='store_true',
                        help="Also save a PR plot per experiment.")
    parser.add_argument('--with-prk-points', action='store_true',
                        help="Overlay mean (Recall@K, Precision@K) point per experiment on the combined plot.")
    args = parser.parse_args()

    # Build experiment list
    if args.experiments and len(args.experiments) > 0:
        exp_subfolders = [EXPERIMENT_FOLDER_FMT.format(bins=b) for b in args.experiments]
    else:
        discovered = discover_experiment_folders()
        if not discovered:
            raise RuntimeError(f"No experiment folders found under '{DESCRIPTOR_FOLDER}'. "
                               f"Expected names like '{EXPERIMENT_FOLDER_FMT.format(bins=12)}'")
        exp_subfolders = [name for _, name in discovered]

    print("Experiments to evaluate:")
    for s in exp_subfolders:
        print(" -", s)

    all_metrics = []
    curve_payloads = []

    for sub in exp_subfolders:
        metrics, curve = evaluate_experiment(
            subfolder=sub,
            topk=args.topk,
            knn=args.knn,
            save_individual_pr=args.save_individual_pr
        )
        all_metrics.append(metrics)
        curve_payloads.append(curve)

    # Save CSV summary
    ensure_dir('plots')
    csv_path = os.path.join('plots', 'requirement2_summary.csv')
    fieldnames = ['experiment', 'bins', 'mAP11', f'Precision@{args.topk}', f'Recall@{args.topk}', 'num_images', 'knn_for_cm']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_metrics, key=lambda r: (r['bins'] if r['bins'] is not None else 0)):
            writer.writerow(row)
    print(f"\nSaved metrics summary: {csv_path}")

    # --- Combined PR plot: ALL experiments on ONE graph ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for payload in sorted(curve_payloads, key=lambda p: (p['bins'] if p['bins'] is not None else 0)):
        ax.plot(payload['recall_grid'],
                payload['mean_precision_curve'],
                marker='o',
                label=payload['label'])
    if args.with_prk_points:
        # Overlay the mean PR@K point for each experiment for quick “top-10” comparison.
        for payload in curve_payloads:
            ax.scatter(payload['mean_R_at_K'],
                       payload['mean_P_at_K'],
                       s=50, marker='s', zorder=5)
            # Annotate lightly with bin count near the point
            if payload['bins'] is not None:
                ax.annotate(f"{payload['bins']}b",
                            (payload['mean_R_at_K'], payload['mean_P_at_K']),
                            textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'PR Curves — All Experiments (P@{args.topk} points shown if enabled)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9, ncol=1)
    plt.tight_layout()
    combined_path = os.path.join('plots', 'pr_curves_all_experiments.png')
    plt.savefig(combined_path, dpi=150)
    print(f"Saved combined PR curves: {combined_path}")
    plt.show()

if __name__ == '__main__':
    main()
