"""
Utility functions for MIRNet training and evaluation
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def set_seed(seed=42):
    """Set random seeds for reproducibility - CPU only"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss function
    More robust than MSE loss, less sensitive to outliers
    """

    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Compute Charbonnier loss

        Args:
            y_pred (torch.Tensor): Predicted images
            y_true (torch.Tensor): Ground truth images

        Returns:
            torch.Tensor: Computed loss
        """
        diff = y_pred - y_true
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


def peak_signal_noise_ratio(y_pred, y_true, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)

    Args:
        y_pred (torch.Tensor): Predicted images [B, C, H, W]
        y_true (torch.Tensor): Ground truth images [B, C, H, W]
        max_val (float): Maximum possible pixel value (1.0 for normalized images)

    Returns:
        torch.Tensor: PSNR value
    """
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def structural_similarity_index(y_pred, y_true, max_val=1.0):
    """
    Calculate Structural Similarity Index (SSIM) - simplified version

    Args:
        y_pred (torch.Tensor): Predicted images [B, C, H, W]
        y_true (torch.Tensor): Ground truth images [B, C, H, W]
        max_val (float): Maximum possible pixel value

    Returns:
        torch.Tensor: SSIM value
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    if y_pred.shape[1] == 3:
        y_pred = torch.mean(y_pred, dim=1, keepdim=True)
        y_true = torch.mean(y_true, dim=1, keepdim=True)

    mu_pred = torch.mean(y_pred)
    mu_true = torch.mean(y_true)

    var_pred = torch.var(y_pred)
    var_true = torch.var(y_true)
    cov = torch.mean((y_pred - mu_pred) * (y_true - mu_true))

    numerator = (2 * mu_pred * mu_true + C1) * (2 * cov + C2)
    denominator = (mu_pred**2 + mu_true**2 + C1) * (var_pred + var_true + C2)

    ssim = numerator / denominator
    return ssim


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath, device):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        filepath: Path to checkpoint file
        device: Device to load model on

    Returns:
        int: Starting epoch
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found: {filepath}")
        return 0

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded: {filepath}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.6f}")

    return epoch + 1


def tensor_to_pil(tensor):
    """
    Convert a tensor to PIL Image

    Args:
        tensor (torch.Tensor): Image tensor [C, H, W] with values in [0, 1]

    Returns:
        PIL.Image: PIL Image
    """
    tensor = torch.clamp(tensor, 0, 1)

    return TF.to_pil_image(tensor)


def plot_results(images, titles, figure_size=(12, 8), save_path=None):
    """
    Plot comparison results

    Args:
        images (list): List of PIL Images or tensors
        titles (list): List of titles for each image
        figure_size (tuple): Figure size
        save_path (str): Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, len(images), figsize=figure_size)

    if len(images) == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        if torch.is_tensor(img):
            img = tensor_to_pil(img.cpu())

        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")

    plt.show()


def infer_single_image(model, image_path, device, save_path=None):
    """
    Perform inference on a single image

    Args:
        model: Trained MIRNet model
        image_path (str): Path to input image
        device: Device to run inference on
        save_path (str): Path to save enhanced image (optional)

    Returns:
        tuple: (original_image, enhanced_image) as PIL Images
    """
    model.eval()

    original_image = Image.open(image_path).convert("RGB")
    input_tensor = TF.to_tensor(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = model(input_tensor)

    enhanced_tensor = enhanced_tensor.squeeze(0).cpu()
    enhanced_image = tensor_to_pil(enhanced_tensor)

    if save_path:
        enhanced_image.save(save_path)
        print(f"Enhanced image saved: {save_path}")

    return original_image, enhanced_image


def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluate model on validation/test data

    Args:
        model: MIRNet model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for low_images, enhanced_images in data_loader:
            low_images = low_images.to(device)
            enhanced_images = enhanced_images.to(device)

            outputs = model(low_images)

            if criterion:
                loss = criterion(outputs, enhanced_images)
                loss_meter.update(loss.item(), low_images.size(0))

            psnr = peak_signal_noise_ratio(outputs, enhanced_images)
            psnr_meter.update(psnr.item(), low_images.size(0))

            ssim = structural_similarity_index(outputs, enhanced_images)
            ssim_meter.update(ssim.item(), low_images.size(0))

    results = {"psnr": psnr_meter.avg, "ssim": ssim_meter.avg}

    if criterion:
        results["loss"] = loss_meter.avg

    return results


if __name__ == "__main__":
    print("Testing utility functions...")

    criterion = CharbonnierLoss()
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    loss = criterion(pred, target)
    print(f"Charbonnier Loss: {loss.item():.6f}")

    psnr = peak_signal_noise_ratio(pred, target)
    print(f"PSNR: {psnr.item():.2f} dB")

    ssim = structural_similarity_index(pred, target)
    print(f"SSIM: {ssim.item():.4f}")

    print("All tests passed!")
