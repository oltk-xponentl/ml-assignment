import random
from typing import List, Tuple

import numpy as np
import deeplake
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ECSSDDataset(Dataset):
    """Wraps the Deep Lake ECSSD dataset for PyTorch.

    It takes a Deep Lake dataset object and a list of indices, applies
    joint transforms on image and mask, and returns:
      - image: float tensor [3, H, W] in [0, 1]
      - mask: float tensor [1, H, W] with values 0 or 1
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

    def _to_pil_image(self, arr: np.ndarray) -> Image.Image:
        """Convert a Deep Lake numpy array to a PIL image.

        Handles both HWC and CHW formats gracefully.
        """
        arr = np.asarray(arr)

        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ds_idx = self.indices[idx]

        sample = self.ds[ds_idx]
        img_arr = sample["images"].numpy()
        mask_arr = sample["masks"].numpy()

        image = self._to_pil_image(img_arr)   # RGB image
        mask = self._to_pil_image(mask_arr)   # grayscale mask

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

    The root_dir and subdir arguments are kept for compatibility with
    other datasets, but are ignored for now. Everything is streamed from hub://activeloop/ecssd
    """

    print("Loading ECSSD from Deep Lake hub://activeloop/ecssd")
    ds = deeplake.load("hub://activeloop/ecssd", read_only=True)

    n = len(ds)
    if n == 0:
        raise RuntimeError("Deep Lake ECSSD dataset returned zero samples")

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    print(f"Total samples {n}")
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
