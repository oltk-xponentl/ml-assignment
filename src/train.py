import argparse
import time
import json
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from data_loader import create_dataloaders
from sod_model import create_model
from metrics import iou_loss, compute_iou, compute_batch_metrics

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    total_loss, total_iou, num_samples = 0.0, 0.0, 0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = bce_loss_fn(logits, masks) + 0.5 * iou_loss(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += compute_iou(logits.detach(), masks.detach()) * batch_size
        num_samples += batch_size

    return {"loss": total_loss / num_samples, "iou": total_iou / num_samples}

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    total_loss, total_iou, num_samples = 0.0, 0.0, 0

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = bce_loss_fn(logits, masks) + 0.5 * iou_loss(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += compute_iou(logits, masks) * batch_size
        num_samples += batch_size

    return {"loss": total_loss / num_samples, "iou": total_iou / num_samples}

# --- CHECKPOINT HELPERS ---
def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0) + 1, ckpt.get("best_val_loss", float("inf"))

def main():
    parser = argparse.ArgumentParser(description="Train SOD model")
    # Config
    parser.add_argument("--data-root", type=str, default="data/ecssd")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    
    # Versioning & Experiments
    parser.add_argument("--exp-name", type=str, required=True, help="e.g. v1_baseline")
    parser.add_argument("--use-bn", action="store_true", help="Add Batch Normalization")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Experiment: {args.exp_name}")

    # 1. Setup Directories
    exp_dir = Path(__file__).resolve().parents[1] / "experiments" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    best_ckpt_path = exp_dir / "best_model.pt"
    last_ckpt_path = exp_dir / "last_checkpoint.pt" 

    # 2. Data & Model
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_root, size=args.size, batch_size=args.batch_size
    )
    model = create_model(use_bn=args.use_bn).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 3. Resume Logic or Start Fresh
    start_epoch = 0
    best_val_loss = float("inf")
    no_improve = 0

    if args.resume and last_ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(last_ckpt_path, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")
    
    # 4. Training Loop
    start_total = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = eval_one_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss {train_stats['loss']:.4f}, IoU {train_stats['iou']:.4f} | "
              f"Val Loss {val_stats['loss']:.4f}, IoU {val_stats['iou']:.4f}")

        # SAVE LAST CHECKPOINT (Every Epoch) 
        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val_loss)

        # SAVE BEST MODEL (Only on improvement)
        if val_stats["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss"]
            no_improve = 0
            # Save just weights for the 'best' model to keep it light for inference
            torch.save(model.state_dict(), best_ckpt_path) 
            print(f"   New best model saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping triggered.")
                break
    
    duration = str(datetime.timedelta(seconds=int(time.time() - start_total)))
    print(f"\nTraining finished in {duration}")

    # 5. Final Evaluation & JSON Report
    if best_ckpt_path.exists():
        print("Running final test evaluation...")
        # Load the best weights for evaluation
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.eval()
        
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "mae": 0.0}
        count = 0
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Test"):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                batch_res = compute_batch_metrics(logits, masks)
                for k in metrics: metrics[k] += batch_res[k]
                count += 1
        
        # Average metrics
        final_metrics = {k: round(v / count, 4) for k, v in metrics.items()}
        
        # Add metadata
        final_metrics["train_duration"] = duration
        final_metrics["epochs_trained"] = epoch + 1
        final_metrics["best_val_loss"] = round(best_val_loss, 4)

        # Save to JSON
        json_path = exp_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"Metrics saved to {json_path}")
        print(final_metrics)

if __name__ == "__main__":
    main()