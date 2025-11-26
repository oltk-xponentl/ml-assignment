import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from sod_model import create_model
from metrics import compute_batch_metrics


def visualize_predictions(
    model,
    data_loader,
    device,
    output_dir: Path,
    num_batches: int = 3,
):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_done = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            for i in range(images.size(0)):
                img = images[i].cpu()
                gt = masks[i].cpu()
                pred = preds[i].cpu()

                img_vis = img.clone()  

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(img_vis.permute(1, 2, 0).numpy())
                axes[0].set_title("Input")
                axes[0].axis("off")

                axes[1].imshow(gt.squeeze().numpy(), cmap="gray")
                axes[1].set_title("GT mask")
                axes[1].axis("off")

                axes[2].imshow(pred.squeeze().numpy(), cmap="gray")
                axes[2].set_title("Pred mask")
                axes[2].axis("off")

                axes[3].imshow(img_vis.permute(1, 2, 0).numpy())
                axes[3].imshow(pred.squeeze().numpy(), cmap="jet", alpha=0.4)
                axes[3].set_title("Overlay")
                axes[3].axis("off")

                out_path = output_dir / f"batch{batch_idx}_sample{i}.png"
                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)

            batches_done += 1
            if batches_done >= num_batches:
                break


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOD model on test set")
    parser.add_argument("--dataset", type=str, default="ecssd", help="ecssd or duts")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--use-skip", action="store_true")
    parser.add_argument("--deep", action="store_true")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Testing on Dataset: {args.dataset}")

    # Pass dataset_name to loader
    _, _, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        size=args.size,
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = create_model(use_bn=args.use_bn, use_skip=args.use_skip, deep=args.deep).to(device)
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print("Evaluating on test set...")
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

    if args.visualize:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "eval_samples"
        print(f"Saving visualizations to {output_dir}")
        visualize_predictions(model, test_loader, device, output_dir)


if __name__ == "__main__":
    main()
