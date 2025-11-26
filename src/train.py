import argparse
import time
import json
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from sod_model import create_model
from metrics import iou_loss, compute_iou, compute_batch_metrics

def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    total_loss, total_iou, num_samples = 0.0, 0.0, 0
    
    use_amp = (scaler is not None)

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = bce_loss_fn(logits, masks) + 0.5 * iou_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = bce_loss_fn(logits, masks) + 0.5 * iou_loss(logits, masks)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        with torch.no_grad():
            total_iou += compute_iou(logits.detach(), masks.detach()) * batch_size
        num_samples += batch_size

    return {"loss": total_loss / num_samples, "iou": total_iou / num_samples}

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    total_loss, total_iou, num_samples = 0.0, 0.0, 0
    
    use_amp = (device.type == 'cuda')

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = bce_loss_fn(logits, masks) + 0.5 * iou_loss(logits, masks)
        else:
            logits = model(images)
            loss = bce_loss_fn(logits, masks) + 0.5 * iou_loss(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += compute_iou(logits, masks) * batch_size
        num_samples += batch_size

    return {"loss": total_loss / num_samples, "iou": total_iou / num_samples}

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
    parser.add_argument("--dataset", type=str, default="ecssd", help="ecssd or duts")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--patience", type=int, default=10)
    
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--use-skip", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Experiment: {args.exp_name}")

    exp_dir = Path(__file__).resolve().parents[1] / "experiments" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    best_ckpt_path = exp_dir / "best_model.pt"
    last_ckpt_path = exp_dir / "last_checkpoint.pt"
    history_path = exp_dir / "history.json" 

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset, 
        size=args.size, 
        batch_size=args.batch_size
    )
    model = create_model(use_bn=args.use_bn, use_skip=args.use_skip, deep=args.deep).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    scaler = None
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed Precision (AMP) Enabled.")
    else:
        print("Running in Standard Precision (CPU mode).")

    start_epoch = 0
    best_val_loss = float("inf")
    no_improve = 0
    
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    if args.resume and last_ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(last_ckpt_path, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
    
    start_total = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_stats = eval_one_epoch(model, val_loader, device)
        
        scheduler.step(val_stats["loss"])
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss {train_stats['loss']:.4f}, IoU {train_stats['iou']:.4f} | "
              f"Val Loss {val_stats['loss']:.4f}, IoU {val_stats['iou']:.4f} | "
              f"LR: {current_lr:.1e}")

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_iou"].append(train_stats["iou"])
        history["val_iou"].append(val_stats["iou"])
        
        with open(history_path, "w") as f:
            json.dump(history, f)

        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val_loss)

        if val_stats["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss"]
            no_improve = 0
            
            # Saving dictionary wrapper for consistency 
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, best_ckpt_path) 
            
            print(f"   New best model saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping triggered.")
                break
    
    duration = str(datetime.timedelta(seconds=int(time.time() - start_total)))
    print(f"\nTraining finished in {duration}")

    if len(history["train_loss"]) > 0:
        graph_path = exp_dir / f"{args.exp_name}_graph.png"
        plt.figure(figsize=(10, 6))
        plt.plot(history["train_loss"], label="Train Loss", color="red")
        plt.plot(history["val_loss"], label="Val Loss", color="blue")
        plt.title(f"Loss Curve: {args.exp_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(graph_path)
        plt.close()
        print(f"Graph saved to {graph_path}")

    if best_ckpt_path.exists():
        print("Running final test evaluation...")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "mae": 0.0}
        count = 0
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Test"):
                images, masks = images.to(device), masks.to(device)
                if device.type == 'cuda':
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                else:
                    logits = model(images)
                batch_res = compute_batch_metrics(logits, masks)
                for k in metrics: metrics[k] += batch_res[k]
                count += 1
        
        final_metrics = {k: round(v / count, 4) for k, v in metrics.items()}
        final_metrics["train_duration"] = duration
        final_metrics["epochs_trained"] = epoch + 1
        final_metrics["best_val_loss"] = round(best_val_loss, 4)

        json_path = exp_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"Metrics saved to {json_path}")
        print(final_metrics)

if __name__ == "__main__":
    main()