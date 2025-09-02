import os
import random
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


class FireClassificationDataset(Dataset):
    """
    Generic binary fire/no-fire dataset loader using YOLO-style label files.

    Labeling rules (binary):
        - D-Fire (dataset_type="real"): class_id == 1 → fire, else no fire
        - Yunnan synthetic (dataset_type="synthetic"): class_id == 0 → fire, else no fire
        - PLOS ONE (dataset_type="plos"): class_id == 0 → fire, else no fire
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        dataset_type: str = "real",
    ) -> None:
        """
        Args:
            image_dir: Path to the folder containing .jpg images.
            label_dir: Path to the folder containing YOLO .txt labels.
            transform: Optional torchvision transforms to apply to the image.
            dataset_type: One of {"real", "synthetic", "plos"} determining the
                          label rule to apply.
        """
        assert dataset_type in {"real", "synthetic", "plos"}, "Invalid dataset_type"
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.dataset_type = dataset_type

        # List all .jpg image files in the folder (sorted for stability)
        self.image_files: List[str] = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith(".jpg")
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = 0
        label_path = os.path.join(self.label_dir, image_name.replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if (
                        (self.dataset_type == "real" and class_id == 1)
                        or (self.dataset_type == "synthetic" and class_id == 0)
                        or (self.dataset_type == "plos" and class_id == 0)
                    ):
                        label = 1
                        break

        return image, torch.tensor(label, dtype=torch.long)


class FireClassificationMixedDataset(Dataset):
    """
    Mixed-source dataset with a fixed total size and a target synthetic ratio.

    Samples without replacement from real and synthetic pools and shuffles the
    combined list for each instantiation.
    """

    def __init__(
        self,
        real_image_dir: str,
        real_label_dir: str,
        syn_image_dir: str,
        syn_label_dir: str,
        syn_ratio: float = 0.5,
        total_samples: int = 5260,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            real_image_dir: Path to real images.
            real_label_dir: Path to real labels.
            syn_image_dir: Path to synthetic images.
            syn_label_dir: Path to synthetic labels.
            syn_ratio: Proportion of synthetic images in the total dataset
                       (e.g., 0.5 for 50%).
            total_samples: Total number of images to include (e.g., 5260).
            transform: Optional torchvision transforms.
        """
        self.real_image_dir = real_image_dir
        self.real_label_dir = real_label_dir
        self.syn_image_dir = syn_image_dir
        self.syn_label_dir = syn_label_dir
        self.transform = transform
        self.syn_ratio = syn_ratio
        self.total_samples = total_samples

        # Available files (sorted for reproducibility before sampling)
        self.real_images: List[str] = sorted(
            f for f in os.listdir(real_image_dir) if f.endswith(".jpg")
        )
        self.syn_images: List[str] = sorted(
            f for f in os.listdir(syn_image_dir) if f.endswith(".jpg")
        )

        # Determine target counts for each source
        target_syn = int(total_samples * syn_ratio)
        target_real = total_samples - target_syn

        # Safety checks: trim target size if not enough images
        target_syn = min(target_syn, len(self.syn_images))
        target_real = min(target_real, len(self.real_images))
        self.actual_total = target_syn + target_real  # may be < total_samples

        # Sample without replacement
        self.sampled_syn_images: List[str] = random.sample(self.syn_images, target_syn)
        self.sampled_real_images: List[str] = random.sample(
            self.real_images, target_real
        )

        # Label data source and combine
        self.data: List[Tuple[str, str]] = (
            [(f, "syn") for f in self.sampled_syn_images]
            + [(f, "real") for f in self.sampled_real_images]
        )
        random.shuffle(self.data)

    def __len__(self) -> int:
        return self.actual_total

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_name, source = self.data[idx]

        if source == "real":
            image_path = os.path.join(self.real_image_dir, image_name)
            label_path = os.path.join(
                self.real_label_dir, image_name.replace(".jpg", ".txt")
            )
        else:
            image_path = os.path.join(self.syn_image_dir, image_name)
            label_path = os.path.join(
                self.syn_label_dir, image_name.replace(".jpg", ".txt")
            )

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = 0
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if (source == "real" and class_id == 1) or (
                        source == "syn" and class_id == 0
                    ):
                        label = 1
                        break

        return image, torch.tensor(label, dtype=torch.long)


class FireClassificationMaskDataset(Dataset):
    """
    Loader for SYN-FIRE (images + masks in PNG).

    Turns a segmentation mask into a binary label:
        - 1 (fire) if the mask has any non-zero pixel
        - 0 (no fire) if the mask is fully black or missing
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Keep only .png images and sort for reproducibility
        self.image_files: List[str] = sorted(
            f for f in os.listdir(self.image_dir) if f.lower().endswith(".png")
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Mask shares the same stem name as the image (e.g., 000123.png)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Default label = 0 (no fire)
        label = 0

        # If a mask exists, check whether any pixel > 0
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")  # grayscale
            # getextrema() returns (min_pixel_value, max_pixel_value)
            _, max_val = mask.getextrema()
            label = 1 if max_val > 0 else 0

        return image, torch.tensor(label, dtype=torch.long)


class FireClassificationPhase4MixedFixed(Dataset):
    """
    Phase 4 helper: fixed-count indoor mixed dataset composed of:
        - PLOS ONE (real) via FireClassificationDataset(dataset_type="plos")
        - SYN-FIRE (synthetic positives) via FireClassificationMaskDataset

    Samples `n_real` from PLOS and `n_syn` from SYN-FIRE without replacement,
    shuffles deterministically, and returns (image, label).
    """

    def __init__(
        self,
        real_image_dir: str,
        real_label_dir: str,
        syn_image_dir: str,
        syn_mask_dir: str,
        n_real: int = 2000,
        n_syn: int = 2000,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        self.transform = transform
        self.seed = seed

        # Reuse existing helpers (keeps label rules consistent)
        self.real_ds = FireClassificationDataset(
            image_dir=real_image_dir,
            label_dir=real_label_dir,
            transform=transform,
            dataset_type="plos",  # PLOS rule: class_id == 0 → fire
        )
        self.syn_ds = FireClassificationMaskDataset(
            image_dir=syn_image_dir,
            mask_dir=syn_mask_dir,
            transform=transform,
        )

        # Build deterministic samples
        rng = random.Random(seed)
        real_count = min(n_real, len(self.real_ds))
        syn_count = min(n_syn, len(self.syn_ds))

        self.real_indices: List[int] = rng.sample(range(len(self.real_ds)), real_count)
        self.syn_indices: List[int] = rng.sample(range(len(self.syn_ds)), syn_count)

        # Tag sources and shuffle combined list
        self._items: List[Tuple[str, int]] = (
            [("real", i) for i in self.real_indices]
            + [("syn", j) for j in self.syn_indices]
        )
        rng.shuffle(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        src, i = self._items[idx]
        if src == "real":
            # Delegates to FireClassificationDataset (plos mode)
            return self.real_ds[i]
        # Delegates to FireClassificationMaskDataset (mask → label)
        return self.syn_ds[i]
