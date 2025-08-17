"""
Training script for MIRNet model
"""

import os
import time
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import project modules
import config
from model import create_mirnet_model
from data import create_data_loaders, print_dataset_info
from utils import (
    set_seed,
    CharbonnierLoss,
    peak_signal_noise_ratio,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    evaluate_model,
)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train model for one epoch

    Args:
        model: MIRNet model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number

    Returns:
        dict: Training metrics for the epoch
    """
    model.train()

    # Metrics tracking
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")

    for batch_idx, (low_images, enhanced_images) in enumerate(pbar):
        # Move to device - removed non_blocking for CPU
        low_images = low_images.to(device)
        enhanced_images = enhanced_images.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(low_images)

        # Calculate loss
        loss = criterion(outputs, enhanced_images)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate PSNR
        with torch.no_grad():
            psnr = peak_signal_noise_ratio(outputs, enhanced_images)

        # Update metrics
        batch_size = low_images.size(0)
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(psnr.item(), batch_size)

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss_meter.avg:.6f}",
                "PSNR": f"{psnr_meter.avg:.2f}",
                "LR": f'{optimizer.param_groups[0]["lr"]:.2e}',
            }
        )

    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg}


def validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Validate model for one epoch

    Args:
        model: MIRNet model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        dict: Validation metrics for the epoch
    """
    model.eval()

    # Metrics tracking
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    # Progress bar
    pbar = tqdm(val_loader, desc=f"Val {epoch:03d}")

    with torch.no_grad():
        for low_images, enhanced_images in pbar:
            # Move to device - removed non_blocking for CPU
            low_images = low_images.to(device)
            enhanced_images = enhanced_images.to(device)

            # Forward pass
            outputs = model(low_images)

            # Calculate loss and PSNR
            loss = criterion(outputs, enhanced_images)
            psnr = peak_signal_noise_ratio(outputs, enhanced_images)

            # Update metrics
            batch_size = low_images.size(0)
            loss_meter.update(loss.item(), batch_size)
            psnr_meter.update(psnr.item(), batch_size)

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss_meter.avg:.6f}", "PSNR": f"{psnr_meter.avg:.2f}"}
            )

    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg}


def train_model(config):
    """
    Main training function

    Args:
        config: Configuration object
    """
    print("Starting MIRNet Training...")
    print("=" * 60)

    # Set random seed
    set_seed(config.RANDOM_SEED)

    # Device setup - fixed for CPU only
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("\nCreating data loaders...")
    data_loaders = create_data_loaders(config)
    print_dataset_info(data_loaders)

    # Create model
    print("\nCreating model...")
    model = create_mirnet_model(config)
    model = model.to(device)
    model.print_model_summary()

    # Loss function and optimizer
    criterion = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",  # Maximize PSNR
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=1e-7,
    )

    # Load checkpoint if exists
    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )

    # Training history
    train_history = {"loss": [], "psnr": []}
    val_history = {"loss": [], "psnr": []}
    best_psnr = 0.0

    print(f"\nStarting training from epoch {start_epoch}...")
    print("=" * 60)

    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start_time = time.time()

        # Train for one epoch
        train_metrics = train_epoch(
            model, data_loaders["train"], optimizer, criterion, device, epoch + 1
        )

        # Validate for one epoch
        val_metrics = validate_epoch(
            model, data_loaders["val"], criterion, device, epoch + 1
        )

        # Update learning rate scheduler
        scheduler.step(val_metrics["psnr"])

        # Save training history
        train_history["loss"].append(train_metrics["loss"])
        train_history["psnr"].append(train_metrics["psnr"])
        val_history["loss"].append(val_metrics["loss"])
        val_history["psnr"].append(val_metrics["psnr"])

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\nEpoch {epoch + 1:03d}/{config.EPOCHS:03d} - Time: {epoch_time:.2f}s")
        print(
            f"Train Loss: {train_metrics['loss']:.6f}, Train PSNR: {train_metrics['psnr']:.2f}"
        )
        print(
            f"Val Loss: {val_metrics['loss']:.6f}, Val PSNR: {val_metrics['psnr']:.2f}"
        )
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics["loss"], checkpoint_path
        )

        # Save best model
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics["loss"], best_model_path
            )
            print(f"New best PSNR: {best_psnr:.2f} - Model saved!")

        print("=" * 60)

    # Final model save
    final_model_path = config.MODEL_SAVE_PATH
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining completed! Final model saved: {final_model_path}")

    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, data_loaders["test"], device, criterion)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test PSNR: {test_metrics['psnr']:.2f}")
    print(f"Test SSIM: {test_metrics['ssim']:.4f}")

    return {
        "model": model,
        "train_history": train_history,
        "val_history": val_history,
        "test_metrics": test_metrics,
    }


def main():
    """Main function"""
    try:
        # Train the model
        results = train_model(config)
        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Run training
    main()
