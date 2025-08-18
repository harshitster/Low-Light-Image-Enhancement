"""
Data loading and preprocessing for MIRNet training
"""

import os
import random
from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class LOLDataset(Dataset):
    """
    Low-light dataset for image enhancement

    Args:
        low_images (list): List of low-light image paths
        enhanced_images (list): List of corresponding enhanced image paths
        image_size (int): Size for random cropping
        is_training (bool): Whether this is for training (enables data augmentation)
    """

    def __init__(self, low_images, enhanced_images, image_size=128, is_training=True):
        self.low_images = low_images
        self.enhanced_images = enhanced_images
        self.image_size = image_size
        self.is_training = is_training

        assert len(low_images) == len(
            enhanced_images
        ), "Number of low and enhanced images must match"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image = Image.open(self.low_images[idx]).convert("RGB")
        enhanced_image = Image.open(self.enhanced_images[idx]).convert("RGB")

        low_tensor = TF.to_tensor(low_image)
        enhanced_tensor = TF.to_tensor(enhanced_image)

        if self.is_training:
            low_tensor, enhanced_tensor = self._random_crop(low_tensor, enhanced_tensor)

        return low_tensor, enhanced_tensor

    def _random_crop(self, low_image, enhanced_image):
        """Random crop for data augmentation during training"""
        _, h, w = low_image.shape

        top = random.randint(0, h - self.image_size)
        left = random.randint(0, w - self.image_size)

        low_cropped = TF.crop(low_image, top, left, self.image_size, self.image_size)
        enhanced_cropped = TF.crop(
            enhanced_image, top, left, self.image_size, self.image_size
        )

        return low_cropped, enhanced_cropped

    def _augment(self, low_image, enhanced_image):
        """Additional data augmentation (optional)"""
        if random.random() > 0.5:
            low_image = TF.hflip(low_image)
            enhanced_image = TF.hflip(enhanced_image)

        if random.random() > 0.5:
            low_image = TF.vflip(low_image)
            enhanced_image = TF.vflip(enhanced_image)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            low_image = TF.rotate(low_image, angle)
            enhanced_image = TF.rotate(enhanced_image, angle)

        return low_image, enhanced_image


class TestDataset(Dataset):
    """
    Dataset for testing/inference (no cropping, full images)

    Args:
        low_images (list): List of low-light image paths
        enhanced_images (list): List of corresponding enhanced image paths (optional for inference)
    """

    def __init__(self, low_images, enhanced_images=None):
        self.low_images = low_images
        self.enhanced_images = enhanced_images

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image = Image.open(self.low_images[idx]).convert("RGB")
        low_tensor = TF.to_tensor(low_image)

        if self.enhanced_images is not None:
            enhanced_image = Image.open(self.enhanced_images[idx]).convert("RGB")
            enhanced_tensor = TF.to_tensor(enhanced_image)
            return low_tensor, enhanced_tensor
        else:
            return low_tensor


def load_data_paths(config):
    """
    Load file paths for training, validation, and test data

    Args:
        config: Configuration object with data paths

    Returns:
        dict: Dictionary containing data paths for train, val, and test sets
    """
    train_low_images = sorted(glob(config.TRAIN_LOW_PATH))[: config.MAX_TRAIN_IMAGES]
    train_enhanced_images = sorted(glob(config.TRAIN_HIGH_PATH))[
        : config.MAX_TRAIN_IMAGES
    ]

    val_low_images = sorted(glob(config.TRAIN_LOW_PATH))[config.MAX_TRAIN_IMAGES :]
    val_enhanced_images = sorted(glob(config.TRAIN_HIGH_PATH))[
        config.MAX_TRAIN_IMAGES :
    ]

    test_low_images = sorted(glob(config.TEST_LOW_PATH))
    test_enhanced_images = sorted(glob(config.TEST_HIGH_PATH))

    return {
        "train": {"low": train_low_images, "enhanced": train_enhanced_images},
        "val": {"low": val_low_images, "enhanced": val_enhanced_images},
        "test": {"low": test_low_images, "enhanced": test_enhanced_images},
    }


def create_data_loaders(config):
    """
    Create DataLoaders for training, validation, and test sets

    Args:
        config: Configuration object

    Returns:
        dict: Dictionary containing DataLoaders for train, val, and test sets
    """
    random.seed(config.RANDOM_SEED)

    data_paths = load_data_paths(config)

    train_dataset = LOLDataset(
        data_paths["train"]["low"],
        data_paths["train"]["enhanced"],
        image_size=config.IMAGE_SIZE,
        is_training=True,
    )

    val_dataset = LOLDataset(
        data_paths["val"]["low"],
        data_paths["val"]["enhanced"],
        image_size=config.IMAGE_SIZE,
        is_training=True,
    )

    test_dataset = TestDataset(
        data_paths["test"]["low"], data_paths["test"]["enhanced"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def print_dataset_info(data_loaders):
    """Print information about the datasets"""
    print("Dataset Information:")
    print(f"- Training batches: {len(data_loaders['train'])}")
    print(f"- Validation batches: {len(data_loaders['val'])}")
    print(f"- Test samples: {len(data_loaders['test'])}")

    sample_batch = next(iter(data_loaders["train"]))
    print(f"- Training batch shape: {sample_batch[0].shape}")
    print(f"- Target batch shape: {sample_batch[1].shape}")


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    import config

    data_loaders = create_data_loaders(config)
    print_dataset_info(data_loaders)

    print("\nTesting data loading...")
    for i, (low, enhanced) in enumerate(data_loaders["train"]):
        print(f"Batch {i}: Low {low.shape}, Enhanced {enhanced.shape}")
        if i >= 2:
            break
