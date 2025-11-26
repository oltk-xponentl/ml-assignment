# Salient Object Detection (SOD) Model

This repository contains a compact implementation and demo for a Salient Object Detection (SOD) model, implemented from scratch.
It includes a U-Net-style model, data loading adapters (Deep Lake for ECSSD, FiftyOne for DUTS), training and evaluation pipelines, and a Streamlit-based interactive demo UI.

**This README covers:** installation, architecture overview, how components work together, running the app/demo, training and evaluation commands, dataset notes, and troubleshooting tips.

**Repository layout**
- `app.py` — Streamlit demo to load trained checkpoints and run inference on uploaded images.
- `src/` — project code
	- `src/sod_model.py` — U-Net style model definition with parameterizable components (batch norm, skip connections, depth)
	- `src/data_loader.py` — `SODDataset` class for unified data handling and adapters for ECSSD (Deep Lake) and DUTS (local or FiftyOne fallback)
	- `src/train.py` — main training loop with mixed precision, learning rate scheduling, checkpointing, and history tracking
	- `src/evaluate.py` — evaluation script with batch metrics computation and optional visualization saving
	- `src/metrics.py` — differentiable IoU loss and batch metric computations (precision, recall, F1, MAE)
- `experiments/` — experiment output folders (checkpoints, metrics, history, graphs)
- `requirements.txt` — all required packages for training, evaluation, and demo

## **Dependencies**

All required packages are listed in `requirements.txt`:

```
torch==2.9.1
torchvision==0.24.1
torchmetrics==1.8.2
numpy==2.2.6
pillow==10.4.0
pandas==2.3.3
matplotlib==3.10.7
tqdm==4.67.1
streamlit==1.51.0
deeplake==3.9.52
fiftyone==1.10.0
fiftyone-brain==0.21.4
fiftyone_db==1.4.0
huggingface_hub==1.1.5
```

All of these packages are mandatory:
- `torch`, `torchvision`: Deep learning framework and vision utilities (model, training, data transforms)
- `numpy`: Array operations
- `pillow`: Image I/O
- `pandas`, `matplotlib`: History tracking, plotting, and visualization
- `tqdm`: Progress bars during training and evaluation
- `streamlit`: Interactive web UI for the demo app
- `deeplake`, `fiftyone`, `huggingface_hub`: Dataset adapters for ECSSD and DUTS datasets
- `torchmetrics`: Metric computation helpers

Notes on CPU vs GPU and torch versions
- `torch` and `torchvision` are the heaviest dependencies. They must match your CUDA toolkit version if you want GPU support.
- If you need CUDA-enabled wheels, install the appropriate `torch`/`torchvision` from https://pytorch.org/get-started/locally/ before installing the rest of the packages.
- The pinned versions above match the original environment; you may relax patch versions (e.g., `2.9.1` to `2.9.x`) if necessary, but ensure `torch` and `torchvision` are compatible with each other and your CUDA version.

## **Installation**

Create a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows (Git Bash or WSL): source .venv/Scripts/activate
```

Install all requirements:

```bash
pip install -r requirements.txt
```

For GPU support, first install the appropriate `torch` and `torchvision` wheels from the official PyTorch website. Example for CUDA 11.8 (adjust version to match your CUDA installation):

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.9.1 torchvision==0.24.1
# then install the rest
pip install torchmetrics numpy pillow pandas matplotlib tqdm streamlit deeplake fiftyone fiftyone-brain fiftyone_db huggingface_hub
```

## **Architecture and how components work together**

This project is structured as an end-to-end pipeline for salient object detection:

1. **Data Loading Pipeline** (`src/data_loader.py`)
   - The `create_dataloaders()` function fetches datasets from remote sources (Deep Lake or FiftyOne) and splits them into train (70%), validation (15%), and test (15%) sets.
   - For ECSSD, the loader connects to activeloop's Deep Lake hub and streams image-mask pairs.
   - For DUTS, the loader first checks for local files (`DUTS-TR-Image`, `DUTS-TR-Mask` folders); if not found, it fetches from FiftyOne's Hugging Face hub.
     - For the purposes of this project, the split is done on the `DUTS-TR` training set, which contains 10,553 images.
        - So, the split was ~7000 training, ~1500 eval, ~1500 test.
     - DUTS also contains a `DUTS-TE` testing set, with around 5000 images. 
   - The `SODDataset` class handles:
     - Loading images and masks from file paths or array-like objects
     - Augmentation for training (rotation, flips, color jitter, random resized crops)
     - Normalization using ImageNet statistics
     - Ensuring masks are strictly binary (0.0 or 1.0)
   - Hugging Face authentication is attempted automatically via `setup_huggingface_auth()` using environment variables or Colab secrets.

2. **Model Architecture** (`src/sod_model.py`)
   - The `SODModel` is a U-Net-style encoder-decoder network designed for binary segmentation.
   - **Encoder blocks** progressively downsample spatial dimensions via max-pooling while increasing feature channels.
   - **Bottleneck** layer processes the lowest resolution feature map.
   - **Decoder blocks** progressively upsample using transposed convolutions and optionally concatenate skip connections from the encoder.
   - **Configuration flags** allow easy experimentation:
     - `use_bn=True`: adds batch normalization after convolutions (helps with training stability)
     - `use_skip=True`: concatenates encoder feature maps to decoder inputs (improves detail preservation)
     - `deep=True`: adds a fourth encoder/decoder layer (increases model capacity and receptive field)
   - Output is a single-channel logit map that is thresholded at 0.5 during inference.

3. **Training Pipeline** (`src/train.py`)
   - The training script accepts command-line arguments to configure the experiment (dataset, model architecture, learning rate, etc.).
   - Each training run creates an experiment folder under `experiments/<exp-name>/` where all outputs are saved.
   - **Loss function**: combines BCE (binary cross-entropy) with differentiable IoU loss weighted at 0.5.
   - **Optimization**: Adam optimizer with weight decay (1e-4) and ReduceLROnPlateau learning rate scheduler.
   - **Mixed precision**: if GPU is available, automatic mixed precision (AMP) is enabled to speed up training and reduce memory usage.
   - **Checkpointing**: best model (lowest validation loss) and latest checkpoint are saved; training can be resumed from the latest checkpoint.
   - **History tracking**: per-epoch metrics (train/validation loss and IoU) are saved to `history.json` and a loss curve is plotted to `<exp-name>_graph.png`.

4. **Evaluation and Metrics** (`src/evaluate.py`, `src/metrics.py`)
   - The `compute_batch_metrics()` function computes precision, recall, F1-score, IoU, and mean absolute error (MAE) by thresholding logits at 0.5.
   - The `evaluate.py` script runs inference on the test set and optionally saves visualization comparisons (input, ground truth, prediction, overlay).
   - Final metrics are printed and can be used to compare model variants.

5. **Interactive Demo** (`app.py`)
   - The Streamlit app loads pre-trained checkpoints from the `experiments/` folder.
   - Users can select a model version from a dropdown, upload an image, and get real-time inference results.
   - The app displays:
     - Model performance metrics (IoU, F1, precision, recall, MAE)
     - Training curves (loss and IoU over epochs) from `history.json`
     - Live inference: original image, predicted mask, and overlay visualization
   - Model loading handles backward compatibility by attempting to map old checkpoint formats to the current model definition.

## **Workflow**

1. **Data acquisition**: `create_dataloaders()` fetches ECSSD or DUTS, caches it locally via Deep Lake/FiftyOne, and splits it.
2. **Model creation**: `create_model()` instantiates a U-Net with chosen architectural flags.
3. **Training**: the training loop processes batches, computes loss, updates weights, and tracks metrics every epoch.
4. **Checkpointing**: best model is saved; if training is interrupted, resume with `--resume` flag.
5. **Evaluation**: final test metrics are computed and saved to `metrics.json`.
6. **Demo**: Streamlit app loads the best checkpoint and serves an interactive interface for inference.

Usage

1) Run the Streamlit demo (interactive model browser and inference)

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (typically `http://localhost:8501`). Use the sidebar to:
- Select a trained model version
- Upload an image
- View live inference results and model performance metrics

2) Train a model

From the project root, run:

```bash
python src/train.py --exp-name my_experiment --dataset ecssd --epochs 25 --batch-size 16 --size 320
```

Key command-line arguments in `src/train.py`:
- `--exp-name` (required): name of the experiment folder under `experiments/` where checkpoints, metrics, and history are saved.
- `--dataset`: `ecssd` (default) or `duts`. ECSSD loads from Deep Lake; DUTS checks for local files first, then falls back to FiftyOne.
- `--epochs`: number of training epochs (default: 25).
- `--batch-size`: batch size for training (default: 16).
- `--lr`: initial learning rate for Adam optimizer (default: 1e-3).
- `--size`: input image resolution (default: 320). All images are resized to this square size.
- `--patience`: early stopping patience for learning rate scheduler (default: 10). If validation loss does not improve for this many steps, the model is stopped. 
    - **NOTE**: Every 5 epochs that validation loss doesn't improve, the learning rate is divided by 10.
- `--use-bn`: enable batch normalization in encoder/decoder blocks (off by default).
- `--use-skip`: enable skip connections from encoder to decoder (off by default).
- `--deep`: add a fourth encoder/decoder layer for increased model depth (off by default).
- `--resume`: resume training from the most recent checkpoint in the experiment folder (off by default).

Example: train a deep U-Net with batch normalization on DUTS for 50 epochs:

```bash
python src/train.py --exp-name deep_unet_duts --dataset duts --epochs 50 --batch-size 8 --use-bn --use-skip --deep
```

3) Evaluate a checkpoint

```bash
python src/evaluate.py --checkpoint experiments/my_experiment/best_model.pt --dataset ecssd --visualize
```

Command-line arguments in `src/evaluate.py`:
- `--checkpoint` (required): path to a saved model checkpoint.
- `--dataset`: `ecssd` or `duts` (default: ecssd). Must match the dataset used during training.
- `--size`: input image size (default: 128). Should match the size used during training.
- `--batch-size`: batch size for evaluation (default: 8).
- `--visualize`: if set, saves comparison images (input, ground truth, prediction, overlay) to `outputs/eval_samples/`.
- `--use-bn`, `--use-skip`, `--deep`: model configuration/architecture flags. Must match the model that was trained.

Example: evaluate and visualize:

```bash
python src/evaluate.py --checkpoint experiments/deep_unet_duts/best_model.pt --dataset duts --size 320 --visualize --use-bn --use-skip --deep
```

## **Dataset notes and data loading**

The dataset loading logic in `src/data_loader.py` is designed to be flexible and robust:

- **ECSSD dataset**: loaded via Deep Lake (activeloop hub). The `get_ecssd_samples()` function connects to `hub://activeloop/ecssd` and retrieves image-mask pairs. 

- **DUTS dataset**: loaded with a two-tier fallback strategy:
  1. **Local-first**: if folders `DUTS-TR-Image` and `DUTS-TR-Mask` exist in the current directory, they are used directly. This is the fastest option if you have pre-downloaded the dataset.
  2. **FiftyOne fallback**: if local files are not found, the loader attempts to fetch `Voxel51/DUTS` from the Hugging Face hub via FiftyOne.
     - Requires a Hugging Face token; the token is auto-detected from environment variables (`HF_TOKEN`) or Colab secrets.
     - May cause issues with the token being limited to certain API calls. As such, it is recommended to download the DUTS-TR set locally.

- **Hugging Face authentication**: if you have a gated dataset or encounter authentication issues, set the `HF_TOKEN` environment variable:
  - On Linux/Mac or Windows Git Bash/WSL:
    ```bash
    export HF_TOKEN="<your_hf_token>"
    ```
  - On Windows PowerShell:
    ```powershell
    $env:HF_TOKEN = "<your_hf_token>"
    ```

- **Dataset splitting**: after loading, the full dataset is shuffled and split into train (70%), validation (15%), and test (15%) using a fixed random seed for reproducibility. The split is performed at the sample level before creating DataLoaders.

- **Data augmentation**: the training dataset applies:
  - Random rotation (up to 15 degrees)
  - Random horizontal flips
  - Random color jitter (brightness, contrast, saturation)
  - Random resized crops (scale 0.5–1.0, aspect ratio 0.8–1.2)
  - All augmentations are applied consistently to both image and mask using nearest-neighbor interpolation for masks to preserve sharp edges.

- **Validation and test datasets**: no augmentation is applied; images and masks are simply resized to the target size.

- **Normalization**: all images are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) after conversion to tensors. Masks are converted to binary (0.0 or 1.0).

**Outputs and experiment structure**
Each experiment folder under `experiments/<exp-name>/` is structured as follows:

- `best_model.pt`: the best model checkpoint (lowest validation loss), saved as a PyTorch state dict with metadata (model weights, optimizer state, epoch, best val loss).
- `last_checkpoint.pt`: the most recent checkpoint, used for resuming interrupted training.
- `metrics.json`: final metrics on the test set (precision, recall, F1-score, IoU, MAE).
- `history.json`: per-epoch training/validation loss and IoU values, saved as lists for easy plotting.
- `<exp_name>_graph.png`: a plot of training and validation loss curves.

Example metrics.json:
```json
{
  "precision": 0.8234,
  "recall": 0.7891,
  "f1": 0.8060,
  "iou": 0.6742,
  "mae": 0.0312,
  "train_duration": "0:45:23",
  "epochs_trained": 25,
  "best_val_loss": 0.2341
}
```

Example history.json (excerpt):
```json
{
  "train_loss": [0.5123, 0.4567, 0.4012, ...],
  "val_loss": [0.4891, 0.4234, 0.3998, ...],
  "train_iou": [0.5234, 0.5789, 0.6123, ...],
  "val_iou": [0.4891, 0.5456, 0.5823, ...]
}
```

## **Troubleshooting and common issues**

- **CUDA / cuDNN errors**: if you see errors like "CUDA out of memory" or "cuDNN not found" at import time, verify your `torch` and CUDA versions match. Reinstall `torch`/`torchvision` with the correct wheel for your CUDA version from https://pytorch.org/get-started/locally/.

- **ImportError for deeplake / fiftyone**: if you attempt to load ECSSD or DUTS but these packages are missing, you will see an ImportError. All packages in `requirements.txt` are mandatory; ensure you have run `pip install -r requirements.txt`.

- **Hugging Face authentication fails**: if you see warnings about HF_TOKEN not being found, set the `HF_TOKEN` environment variable before running training/evaluation. Alternatively, the code can still access public datasets but may fail on gated ones.

- **Streamlit import errors**: if the Streamlit app cannot import `src` modules, ensure you run `streamlit run app.py` from the project root directory so Python path resolution works correctly. The app explicitly appends `src` to `sys.path` in the first few lines.

- **Out of memory during training**: reduce `--batch-size` or `--size` (input resolution) to fit your GPU memory. For reference, batch size 16 at 320x320 typically requires ~8GB GPU memory.

- **Slow data loading on first run**: on first run, datasets may be cached/downloaded (especially from Deep Lake or FiftyOne). This is normal; subsequent runs will be faster.

- **Training loss is NaN**: this typically indicates numerical instability. Try reducing the learning rate (`--lr`) or enabling batch normalization (`--use-bn`).

## **Development notes**

- **Model design**: the SODModel is intentionally simple and modular. EncoderBlocks and DecoderBlocks can be easily extended (e.g., adding residual connections, changing activation functions).

- **Loss function**: the combined BCE + IoU loss encourages the model to optimize both pixel-wise accuracy and region-level overlap. The 0.5 weight for IoU loss was empirically chosen but can be tuned.

- **Mixed precision training**: when GPU is detected, automatic mixed precision (AMP) is enabled. This speeds up training by ~1.5x and reduces memory usage by ~20-30% with negligible accuracy loss. For CPU, standard precision is used.

- **Checkpointing strategy**: the training loop saves both the best checkpoint (lowest val loss) and the latest checkpoint. The latest checkpoint includes optimizer state, so training can be resumed exactly where it left off using `--resume`.

- **Learning rate scheduling**: `ReduceLROnPlateau` monitors validation loss and reduces the learning rate by a factor of 0.1 if no improvement is seen for 5 consecutive epochs. This helps fine-tune models in later training stages.

