from typing import Dict

import torch
import torch.nn.functional as F


def iou_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Differentiable IoU loss for binary segmentation.

    Uses probabilities instead of hard thresholding.
    Loss = 1 - mean IoU over the batch.
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    # Flatten over spatial dims, keep batch dim
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = (probs + targets - probs * targets).sum(dim=dims)

    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou.mean()


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute IoU, precision, recall, F1, MAE for a batch."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    # Flatten batch and spatial dimensions
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # True positives, false positives, false negatives
    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1.0 - targets_flat)).sum(dim=1)
    fn = ((1.0 - preds_flat) * targets_flat).sum(dim=1)
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum(dim=1)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    # Mean absolute error on probabilities, not thresholded
    mae = torch.abs(probs.view(probs.size(0), -1) - targets_flat).mean(dim=1)

    metrics = {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
        "mae": mae.mean().item(),
    }
    return metrics


def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Convenience function that returns mean IoU for a batch."""
    return compute_batch_metrics(
        logits=logits, targets=targets, threshold=threshold, eps=eps
    )["iou"]
