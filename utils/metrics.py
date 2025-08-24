import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,   # ⇦ NEW
)

# ------------------------------
# Backwards-compatible basic metrics
# ------------------------------
def calculate_metrics(labels, preds):
    """
    Computes accuracy, precision, recall, and F1 score.

    Args:
        labels (Tensor/ndarray/list): Ground-truth class labels (0/1).
        preds  (Tensor/ndarray/list): Predicted class labels (0/1).

    Returns:
        dict: {accuracy, precision, recall, f1}
    """
    # Accept torch.Tensors or numpy/list; convert to numpy for sklearn
    labels_np = _to_numpy_1d(labels)
    preds_np  = _to_numpy_1d(preds)

    return {
        "accuracy":  accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall":    recall_score(labels_np, preds_np, zero_division=0),
        "f1":        f1_score(labels_np, preds_np, zero_division=0),
    }


# ------------------------------
# New: One-call evaluation + saving
# ------------------------------
def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_prob_pos,
    testset_name,
    drive_base_dir="/content/drive/MyDrive/fire-detection-dissertation",
):
    """
    Compute all metrics, then save:
      - JSON metrics to:   {drive}/results/metrics/<TESTSET>/<model_name>.json
      - Confusion matrix:  {drive}/figures/confusion_matrices/<TESTSET>/<model_name>.png
      - ROC curve:         {drive}/figures/roc_curves/<TESTSET>/<model_name>.png
      - PR curve:          {drive}/figures/pr_curves/<TESTSET>/<model_name>.png
    """
    # --- Canonicalise inputs ---
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)
    y_prob_pos = _to_numpy_1d(y_prob_pos).astype(float)

    testset_key = _canon_testset_name(testset_name)  # -> "D_FIRE" or "PLOS_ONE"

    # --- Output paths (match your project layout) ---
    metrics_dir = os.path.join(drive_base_dir, "results", "metrics", testset_key)
    fig_cm_dir  = os.path.join(drive_base_dir, "figures", "confusion_matrices", testset_key)
    fig_roc_dir = os.path.join(drive_base_dir, "figures", "roc_curves", testset_key)
    fig_pr_dir  = os.path.join(drive_base_dir, "figures", "pr_curves", testset_key)
    for d in (metrics_dir, fig_cm_dir, fig_roc_dir, fig_pr_dir):
        os.makedirs(d, exist_ok=True)

    # --- Confusion counts (labels order fixed as [0,1]) ---
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # --- Scalar metrics (same definitions your notebook used) ---
    accuracy    = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision   = tp / max(1, (tp + fp))  # positive-class precision
    recall      = tp / max(1, (tp + fn))  # TPR / sensitivity
    f1          = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / max(1, (tn + fp))  # TNR
    fpr         = fp / max(1, (fp + tn))
    fnr         = fn / max(1, (fn + tp))
    mcc         = float(matthews_corrcoef(y_true, y_pred)) if _has_both_classes(y_pred) else 0.0

    # Threshold-free summaries
    roc_auc     = float(roc_auc_score(y_true, y_prob_pos))
    pr_auc      = float(average_precision_score(y_true, y_prob_pos))  # AP

    # --- NEW: per-class metrics (separate section) ---
    # labels fixed as [0,1] so index 0 = no_fire, index 1 = fire
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    per_class = {
        "no_fire": {
            "precision": float(prec_arr[0]),
            "recall":    float(rec_arr[0]),
            "f1":        float(f1_arr[0]),
            "support":   int(sup_arr[0]),
        },
        "fire": {
            "precision": float(prec_arr[1]),
            "recall":    float(rec_arr[1]),
            "f1":        float(f1_arr[1]),
            "support":   int(sup_arr[1]),
        },
    }

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": 0.5,  # convention used to produce y_pred
        "per_class": per_class,  # ⇦ NEW: clean, separate section
    }

    # --- Save JSON ---
    json_path = os.path.join(metrics_dir, f"{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # --- Confusion matrix plot (counts) ---
    _plot_confusion_counts(tn, fp, fn, tp,
                           title=f"Confusion Matrix – {model_name}",
                           save_path=os.path.join(fig_cm_dir, f"{model_name}.png"))

    # --- ROC curve ---
    _plot_roc_curve(y_true, y_prob_pos,
                    title=f"ROC – {model_name} (AUC={roc_auc:.3f})",
                    save_path=os.path.join(fig_roc_dir, f"{model_name}.png"))

    # --- PR curve ---
    _plot_pr_curve(y_true, y_prob_pos,
                   title=f"PR – {model_name} (AP={pr_auc:.3f})",
                   save_path=os.path.join(fig_pr_dir, f"{model_name}.png"))

    print(f"✅ Saved metrics and plots for {model_name} on {testset_key}")
    return metrics



# ------------------------------
# Internal helpers
# ------------------------------
def _to_numpy_1d(x):
    """Accept torch.Tensor, list, or ndarray; return 1D np.ndarray."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    return x.reshape(-1)

def _has_both_classes(y_pred_np):
    """Return True if predictions contain both 0 and 1 (avoids undefined MCC edge cases)."""
    u = np.unique(y_pred_np)
    return (0 in u) and (1 in u)

def _canon_testset_name(name):
    """Map a loose name to the canonical folder name used in the repo."""
    n = str(name).strip().lower().replace(" ", "")
    if n in {"dfire", "d_fire", "d-fire", "d"}:
        return "D_FIRE"
    if n in {"plos", "plosone", "plos_one", "indoor"}:
        return "PLOS_ONE"
    raise ValueError(f"Unknown testset_name: {name!r}. Expected 'dfire' or 'plosone'.")

def _plot_confusion_counts(tn, fp, fn, tp, title, save_path):
    cm = np.array([[tn, fp],
                   [fn, tp]])
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="red", fontsize=12)
    plt.xticks([0,1], ["No Fire", "Fire"])
    plt.yticks([0,1], ["No Fire", "Fire"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def _plot_roc_curve(y_true, y_prob_pos, title, save_path):
    fpr_pts, tpr_pts, _ = roc_curve(y_true, y_prob_pos)
    auc_val = roc_auc_score(y_true, y_prob_pos)
    plt.figure()
    plt.plot(fpr_pts, tpr_pts, label=f"AUC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def _plot_pr_curve(y_true, y_prob_pos, title, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
    ap = average_precision_score(y_true, y_prob_pos)
    plt.figure()
    plt.step(recall, precision, where="post", label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
