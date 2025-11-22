import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_loader import create_dataloaders
from model import create_model
from metrics import iou_loss, compute_iou, compute_batch_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    num_samples = 0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)

        bce = bce_loss_fn(logits, masks)
        iou_l = iou_loss(logits, masks)
        loss = bce + 0.5 * iou_l

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += compute_iou(logits.detach(), masks.detach()) * batch_size
        num_samples += batch_size

    avg_loss = total_loss / num_samples
    avg_iou = total_iou / num_samples

    return {"loss": avg_loss, "iou": avg_iou}


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    num_samples = 0

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        bce = bce_loss_fn(logits, masks)
        iou_l = iou_loss(logits, masks)
        loss = bce + 0.5 * iou_l

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += compute_iou(logits, masks) * batch_size
        num_samples += batch_size

    avg_loss = total_loss / num_samples
    avg_iou = total_iou / num_samples
    return {"loss": avg_loss, "iou": avg_iou}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
) -> None:
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train SOD model")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "ecssd"),
        help="Root directory of ECSSD dataset with images and masks folders",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Image size to resize to",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint if available",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs with no val loss improvement before early stop",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_root,
        size=args.size,
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = create_model().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last_checkpoint.pt"
    best_ckpt_path = ckpt_dir / "best_model.pt"

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume and last_ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            last_ckpt_path, model, optimizer
        )

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = eval_one_epoch(model, val_loader, device)

        elapsed = time.time() - start_time
        print(
            f"Train loss {train_stats['loss']:.4f}, IoU {train_stats['iou']:.4f} "
            f"Val loss {val_stats['loss']:.4f}, IoU {val_stats['iou']:.4f} "
            f"Time {elapsed:.1f}s"
        )

        # Save last checkpoint every epoch
        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
        )
        print(f"Saved checkpoint to {last_ckpt_path}")

        # Track best validation loss
        if val_stats["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
            )
            print(f"New best model saved to {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in val loss for {epochs_without_improvement} epochs"
            )
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered")
                break

    # Final evaluation on test set using best checkpoint
    if best_ckpt_path.exists():
        print("\nEvaluating best model on test set")
        # Reload best weights
        state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        model.to(device)

        model.eval()
        all_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "mae": 0.0}
        num_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Test"):
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                batch_metrics = compute_batch_metrics(logits, masks)
                for k in all_metrics:
                    all_metrics[k] += batch_metrics[k]
                num_batches += 1

        for k in all_metrics:
            all_metrics[k] /= max(num_batches, 1)

        print("Test metrics:")
        for k, v in all_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Best checkpoint not found, skipping test evaluation")


if __name__ == "__main__":
    main()
