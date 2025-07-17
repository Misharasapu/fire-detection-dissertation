import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class FireClassificationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing .jpg images.
            label_dir (str): Path to the folder containing YOLO .txt labels.
            transform (callable, optional): Optional torchvision transforms to apply to the image.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # List all .jpg image files in the folder
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.image_files.sort()  # To keep order consistent

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Step 1: Get image filename
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Step 2: Load image
        image = Image.open(image_path).convert("RGB")  # ensure 3-channel RGB

        # Step 3: Apply transform if provided
        if self.transform:
            image = self.transform(image)

        # Step 4: Determine corresponding label file
        label_file = image_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_file)

        # Step 5: Determine binary class (fire = 1, no fire = 0)
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    # ✅ CORRECTED LINE:
                    if class_id == 1:  # fire
                        label = 1
                        break

        return image, torch.tensor(label, dtype=torch.long)



class FireClassificationSyntheticDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Path to folder containing synthetic .jpg images.
            label_dir (str): Path to folder containing YOLO .txt labels.
            transform (callable, optional): torchvision transforms to apply.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # List all .jpg image files in the folder
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.image_files.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_file = image_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_file)

        # 🔁 MSFFD (synthetic) logic: class_id == 0 → fire
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id == 0:  # fire in MSFFD
                        label = 1
                        break

        return image, torch.tensor(label, dtype=torch.long)


class FireClassificationMixedDataset(Dataset):
    def __init__(self, real_image_dir, real_label_dir,
                 syn_image_dir, syn_label_dir,
                 syn_ratio=0.5, total_samples=5261, transform=None):
        """
        Creates a fixed-size dataset by mixing synthetic and real images based on a target ratio.

        Args:
            real_image_dir (str): Path to real images
            real_label_dir (str): Path to real labels
            syn_image_dir (str): Path to synthetic images
            syn_label_dir (str): Path to synthetic labels
            syn_ratio (float): Proportion of synthetic images in the total dataset (e.g. 0.5 for 50%)
            total_samples (int): Total number of images in the dataset (e.g. 5,261)
            transform (callable, optional): torchvision transforms
        """
        self.real_image_dir = real_image_dir
        self.real_label_dir = real_label_dir
        self.syn_image_dir = syn_image_dir
        self.syn_label_dir = syn_label_dir
        self.transform = transform
        self.syn_ratio = syn_ratio
        self.total_samples = total_samples

        # Get available files
        self.real_images = sorted([f for f in os.listdir(real_image_dir) if f.endswith('.jpg')])
        self.syn_images = sorted([f for f in os.listdir(syn_image_dir) if f.endswith('.jpg')])

        # Determine target counts for each source
        target_syn = int(total_samples * syn_ratio)
        target_real = total_samples - target_syn

        # Safety check: trim target size if we don’t have enough images
        target_syn = min(target_syn, len(self.syn_images))
        target_real = min(target_real, len(self.real_images))
        self.actual_total = target_syn + target_real  # may be slightly less than total_samples

        # Sample without replacement
        self.sampled_syn_images = random.sample(self.syn_images, target_syn)
        self.sampled_real_images = random.sample(self.real_images, target_real)

        # Label data source and combine
        self.data = [(f, 'syn') for f in self.sampled_syn_images] + \
                    [(f, 'real') for f in self.sampled_real_images]
        random.shuffle(self.data)

    def __len__(self):
        return self.actual_total

    def __getitem__(self, idx):
        image_name, source = self.data[idx]

        if source == 'real':
            image_path = os.path.join(self.real_image_dir, image_name)
            label_path = os.path.join(self.real_label_dir, image_name.replace('.jpg', '.txt'))
        else:
            image_path = os.path.join(self.syn_image_dir, image_name)
            label_path = os.path.join(self.syn_label_dir, image_name.replace('.jpg', '.txt'))

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if (source == 'real' and class_id == 1) or (source == 'syn' and class_id == 0):
                        label = 1
                        break

        return image, torch.tensor(label, dtype=torch.long)
