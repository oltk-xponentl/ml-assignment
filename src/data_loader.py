import random
import os
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# 1. DeepLake (for ECSSD)
try:
    import deeplake
except ImportError:
    deeplake = None

# 2. FiftyOne (for DUTS)
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    import fiftyone.utils.huggingface as fouh
except ImportError:
    fo = None

# 3. Google Colab Secrets (Only works on Colab)
try:
    from google.colab import userdata
except ImportError:
    userdata = None

# Authentication Helper
def setup_huggingface_auth():
    """
    Securely authenticates with Hugging Face.
    Priority 1: Google Colab Secrets (HF_TOKEN)
    Priority 2: OS Environment Variable (HF_TOKEN)
    """
    token = None
    
    # Try Colab Secrets
    if userdata:
        try:
            token = userdata.get('HF_TOKEN')
        except:
            pass
            
    # Try Environment Variable (Local)
    if not token:
        token = os.getenv('HF_TOKEN')
        
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            import huggingface_hub
            huggingface_hub.login(token=token)
            print("Logged in to Hugging Face successfully.")
        except ImportError:
            print("huggingface_hub not installed. Skipping login.")
    else:
        print("No HF_TOKEN found. Public datasets will work; gated ones might fail.")

# Run auth setup immediately
setup_huggingface_auth()


class SODDataset(Dataset):
    """
    Source-Agnostic Dataset. 
    Expects a list of dictionaries: [{'img': path_or_array, 'mask': path_or_array}, ...]
    """
    def __init__(self, samples: List[Dict], size=224, augment=False):
        self.samples = samples
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # --- Load Image ---
        if isinstance(sample["img"], str):
            image = Image.open(sample["img"]).convert("RGB")
        else:
            # DeepLake returns numpy-like objects
            img_arr = np.asarray(sample["img"])

        # PIL cannot handle (H, W, 1). It needs (H, W) for grayscale or (H, W, 3) for RGB.
            if img_arr.ndim == 3 and img_arr.shape[-1] == 1:
                img_arr = img_arr.squeeze(-1)

            # Normalize 0-1 -> 0-255 if needed
            if img_arr.dtype != np.uint8 and img_arr.max() <= 1.5:
                img_arr = (img_arr * 255.0).astype(np.uint8)
            image = Image.fromarray(img_arr).convert("RGB")

        # --- Load Mask ---
        if isinstance(sample["mask"], str):
            mask = Image.open(sample["mask"]).convert("L")
        else:
            mask_arr = np.asarray(sample["mask"])
            # Fix dimensions if DeepLake adds channel dim
            if mask_arr.ndim == 3: 
                mask_arr = mask_arr[..., 0]
            mask_img = (mask_arr * 255.0).round().clip(0, 255).astype(np.uint8)
            mask = Image.fromarray(mask_img).convert("L")

        # --- Augmentation ---
        if self.augment:
            # 1. Geometric (Flip/Rotate)
            if random.random() < 0.5:
                angle = random.randint(-15, 15)
                image = TF.rotate(image, angle)
                # Nearest Neighbor for Mask Rotation (No blurring)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
                
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 2. Color Jitter (Image Only)
            if random.random() < 0.8:
                if random.random() < 0.5: image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                if random.random() < 0.5: image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                if random.random() < 0.5: image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            
            # 3. RandomResizedCrop
            # scale=(0.5, 1.0) = Zoom in up to 2x
            i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(0.8, 1.2))
            image = TF.resized_crop(image, i, j, h, w, (self.size, self.size))
            # Nearest Neighbor for Mask Resize
            mask = TF.resized_crop(mask, i, j, h, w, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)
            
        else:
            # Validation: Simple Resize
            image = TF.resize(image, (self.size, self.size))
            mask = TF.resize(mask, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)

        # --- Tensor & Normalization ---
        image_t = TF.to_tensor(image)
        mask_t = TF.to_tensor(mask)
        
        # ImageNet Stats Normalization
        image_t = TF.normalize(image_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if mask_t.shape[0] > 1: mask_t = mask_t[:1, ...]
        # Ensure strict binary 0.0 or 1.0
        mask_t = (mask_t > 0.5).float()
        
        return image_t, mask_t

def get_ecssd_samples():
    """Adapter for DeepLake ECSSD"""
    if deeplake is None:
        raise ImportError("deeplake is not installed.")
    print("Loading ECSSD from Deep Lake...")
    ds = deeplake.load("hub://activeloop/ecssd", read_only=True)
    samples = []
    for img, mask in zip(ds.images, ds.masks):
        samples.append({
            "img": img.numpy(), # Convert to numpy immediately to store in list
            "mask": mask.numpy()
        })
    return samples

def get_duts_samples():
    """Adapter for DUTS (Prioritize Local, Fallback to FiftyOne)"""
    
    # OPTION A: Local File Check
    local_img_dir = "DUTS-TR-Image"
    local_mask_dir = "DUTS-TR-Mask"
    
    if os.path.exists(local_img_dir) and os.path.exists(local_mask_dir):
        print(f"Found local DUTS data in {local_img_dir}. using local files.")
        samples = []
        fnames = [f for f in os.listdir(local_img_dir) if f.endswith(".jpg")]
        for fname in fnames:
            img_path = os.path.join(local_img_dir, fname)
            mask_name = fname.replace(".jpg", ".png")
            mask_path = os.path.join(local_mask_dir, mask_name)
            if os.path.exists(mask_path):
                samples.append({"img": img_path, "mask": mask_path})
        return samples

    # OPTION B: FiftyOne Fallback
    if fo is None:
        raise ImportError("fiftyone not installed.")
    
    print("Loading DUTS-TR from Hugging Face via FiftyOne...")
    dataset = fouh.load_from_hub("Voxel51/DUTS", split="train")    
    samples = []
    print("Extracting paths from FiftyOne...")
    
    for sample in dataset:
        img_path = sample.filepath
        
        mask_path = None
        # Attempt to find mask path in FiftyOne metadata
        if sample.ground_truth and hasattr(sample.ground_truth, "mask_path"):
             mask_path = sample.ground_truth.mask_path
        
        # Fallback: DUTS directory structure assumption
        if not mask_path:
            mask_path = img_path.replace("DUTS-TR-Image", "DUTS-TR-Mask").replace(".jpg", ".png")
            
        samples.append({"img": img_path, "mask": mask_path})
        
    return samples

def create_dataloaders(
    dataset_name: str = "ecssd",
    size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
):
    # 1. Select Data Source
    if dataset_name.lower() == "duts":
        samples = get_duts_samples()
    else:
        samples = get_ecssd_samples()

    n = len(samples)
    print(f"Total samples: {n}")

    # 2. Shuffle and Split (70/15/15)
    # We perform this split manually on the source data to ensure consistency
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train : n_train + n_val]]
    test_samples = [samples[i] for i in indices[n_train + n_val :]]

    print(f"Split: Train {len(train_samples)}, Val {len(val_samples)}, Test {len(test_samples)}")

    # 3. Create Dataset Objects
    train_ds = SODDataset(train_samples, size=size, augment=True)
    val_ds = SODDataset(val_samples, size=size, augment=False)
    test_ds = SODDataset(test_samples, size=size, augment=False)

    # 4. Create DataLoaders
    # Pin memory improves transfer speed to GPU
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )