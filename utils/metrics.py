from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(labels, preds):
    """
    Computes accuracy, precision, recall, and F1 score.

    Args:
        labels (Tensor): Ground-truth class labels.
        preds (Tensor): Predicted class labels.

    Returns:
        dict: Dictionary with accuracy, precision, recall, and F1 score.
    """
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    return {
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0)
    }
