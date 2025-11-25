import random
from typing import List, Tuple
import deeplake
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class SODDataset(Dataset):
    def __init__(self, deeplake_ds, indices, size=224, augment=False):
        self.ds = deeplake_ds
        self.indices = indices
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.ds[self.indices[idx]]
        
        # Image
        img_arr = sample["images"].numpy()
        img_arr = np.asarray(img_arr)
        if img_arr.ndim == 3:
            if img_arr.shape[0] in (1, 3) and img_arr.shape[-1] not in (1, 3):
                img_arr = np.transpose(img_arr, (1, 2, 0))
        elif img_arr.ndim == 2:
            img_arr = img_arr[..., None]
        if img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = np.repeat(img_arr, 3, axis=2)
        img_arr = img_arr.astype("float32")
        if img_arr.max() <= 1.5: img_arr *= 255.0
        img_arr = np.clip(img_arr, 0.0, 255.0).astype("uint8")
        image = Image.fromarray(img_arr).convert("RGB")

        # Mask
        mask_arr = sample["masks"].numpy()
        mask_arr = np.asarray(mask_arr)
        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1: mask_arr = mask_arr[..., 0]
        if mask_arr.ndim > 2 and mask_arr.shape[0] == 1: mask_arr = mask_arr[0]
        mask_arr = mask_arr.astype("float32")
        mask_img = (mask_arr * 255.0).round().clip(0, 255).astype("uint8")
        mask = Image.fromarray(mask_img).convert("L")

        # Augmentation
        if self.augment:
            resize_size = int(self.size * 1.15)
            image = TF.resize(image, (resize_size, resize_size))
            mask = TF.resize(mask, (resize_size, resize_size), interpolation=TF.InterpolationMode.NEAREST)
            
            if random.random() < 0.5:
                angle = random.randint(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.8:
                if random.random() < 0.5: image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                if random.random() < 0.5: image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                if random.random() < 0.5: image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.size, self.size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        else:
            image = TF.resize(image, (self.size, self.size))
            mask = TF.resize(mask, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)

        image_t = TF.to_tensor(image)
        mask_t = TF.to_tensor(mask)
        
        # Normalization (Standard ImageNet stats) 
        image_t = TF.normalize(image_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if mask_t.shape[0] > 1: mask_t = mask_t[:1, ...]
        mask_t = (mask_t > 0.5).float()
        return image_t, mask_t

def create_dataloaders(
    dataset_name: str = "ecssd",
    root_dir: str = "",
    size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
):
    if dataset_name.lower() == "duts":
        print("Loading DUTS-TR (10k images) from Deep Lake...")
        url = "hub://activeloop/duts-train"
    else:
        print("Loading ECSSD (1k images) from Deep Lake...")
        url = "hub://activeloop/ecssd"

    ds = deeplake.load(url, read_only=True)
    n = len(ds)
    if n == 0: raise RuntimeError("Dataset empty")
    
    print(f"Total samples: {n}")

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    print(f"Split: Train {len(train_indices)}, Val {len(val_indices)}, Test {len(test_indices)}")

    train_ds = SODDataset(ds, train_indices, size=size, augment=True)
    val_ds = SODDataset(ds, val_indices, size=size, augment=False)
    test_ds = SODDataset(ds, test_indices, size=size, augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )