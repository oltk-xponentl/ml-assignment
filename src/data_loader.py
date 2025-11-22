import random
from typing import List, Tuple

import deeplake
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ECSSDDataset(Dataset):
    """Deep Lake ECSSD dataset wrapper for PyTorch.

    Returns:
      image: float tensor [3, H, W] in [0, 1]
      mask:  float tensor [1, H, W] with values 0 or 1
    """

    def __init__(
        self,
        deeplake_ds: deeplake.Dataset,
        indices: List[int],
        size: int = 128,
        augment: bool = False,
    ) -> None:
        self.ds = deeplake_ds
        self.indices = indices
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds_idx = self.indices[idx]
        sample = self.ds[ds_idx]

        # ---------- IMAGE ----------
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
        i_min, i_max = float(img_arr.min()), float(img_arr.max())
        if i_max <= 1.5:
            img_arr = img_arr * 255.0
        img_arr = np.clip(img_arr, 0.0, 255.0).astype("uint8")

        image = Image.fromarray(img_arr).convert("RGB")

        mask_arr = sample["masks"].numpy()
        mask_arr = np.asarray(mask_arr)

        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim > 2 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]

        mask_arr = mask_arr.astype("float32")  # 0 or 1

        mask_img = (mask_arr * 255.0).round().clip(0, 255).astype("uint8")
        mask = Image.fromarray(mask_img).convert("L")

        target_size = self.size

        if self.augment:
            resize_size = int(target_size * 1.1)
            image = TF.resize(image, (resize_size, resize_size))
            mask = TF.resize(
                mask,
                (resize_size, resize_size),
                interpolation=TF.InterpolationMode.NEAREST,
            )

            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            brightness_factor = 0.8 + 0.4 * random.random()
            image = TF.adjust_brightness(image, brightness_factor)

            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=(target_size, target_size)
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        else:
            image = TF.resize(image, (target_size, target_size))
            mask = TF.resize(
                mask,
                (target_size, target_size),
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image_t = TF.to_tensor(image)     
        mask_t = TF.to_tensor(mask)       

        if mask_t.shape[0] > 1:
            mask_t = mask_t[:1, ...]

        mask_t = (mask_t > 0.5).float()

        return image_t, mask_t


def create_dataloaders(
    root_dir: str = "",
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    size: int = 128,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train, val, test DataLoaders from Deep Lake ECSSD.

    root_dir and subdir args are kept for compatibility but ignored.
    Everything is streamed from hub://activeloop/ecssd.
    """
    print("Loading ECSSD from Deep Lake hub://activeloop/ecssd")
    ds = deeplake.load("hub://activeloop/ecssd", read_only=True)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Deep Lake ECSSD dataset returned zero samples")

    print(f"Total samples {n}")

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    print(f"Train {len(train_indices)}, Val {len(val_indices)}, Test {len(test_indices)}")

    train_ds = ECSSDDataset(ds, train_indices, size=size, augment=True)
    val_ds = ECSSDDataset(ds, val_indices, size=size, augment=False)
    test_ds = ECSSDDataset(ds, test_indices, size=size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(
        size=128,
        batch_size=4,
        num_workers=0,
    )
    images, masks = next(iter(train_loader))
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
    print("Mask min:", masks.min().item(), "max:", masks.max().item())
    print("Mask unique values:", torch.unique(masks))
