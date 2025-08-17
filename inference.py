"""
Inference script for MIRNet model
"""

import os
import random
from glob import glob
import argparse

import torch
import matplotlib.pyplot as plt

# Import project modules
import config
from model import create_mirnet_model
from data import create_data_loaders
from utils import (
    set_seed,
    infer_single_image,
    plot_results,
    tensor_to_pil,
    evaluate_model,
    CharbonnierLoss,
)


def load_trained_model(model_path, device):
    """
    Load trained MIRNet model

    Args:
        model_path (str): Path to saved model
        device: Device to load model on

    Returns:
        MIRNet: Loaded model
    """
    # Create model
    model = create_mirnet_model(config)

    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return None

    model = model.to(device)
    model.eval()

    return model


def test_on_dataset(model, data_loaders, device, num_samples=6):
    """
    Test model on random samples from test dataset

    Args:
        model: Trained MIRNet model
        data_loaders: Dictionary containing data loaders
        device: Device to run inference on
        num_samples (int): Number of random samples to test
    """
    print(f"\nTesting on {num_samples} random samples from test dataset...")

    # Get test data loader
    test_loader = data_loaders["test"]

    # Convert to list for random sampling
    test_data = list(test_loader.dataset)

    # Sample random images
    random_indices = random.sample(
        range(len(test_data)), min(num_samples, len(test_data))
    )

    for i, idx in enumerate(random_indices):
        print(f"\nProcessing sample {i + 1}/{len(random_indices)}")

        # Get data
        if len(test_data[idx]) == 2:  # Has ground truth
            low_tensor, enhanced_tensor = test_data[idx]
        else:  # Only low-light image
            low_tensor = test_data[idx]
            enhanced_tensor = None

        # Move to device and add batch dimension
        low_input = low_tensor.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output_tensor = model(low_input)

        # Remove batch dimension and move to CPU
        output_tensor = output_tensor.squeeze(0).cpu()

        # Convert to PIL images
        low_image = tensor_to_pil(low_tensor)
        enhanced_image = tensor_to_pil(output_tensor)

        # Prepare for plotting
        images = [low_image, enhanced_image]
        titles = ["Original (Low-light)", "MIRNet Enhanced"]

        # Add ground truth if available
        if enhanced_tensor is not None:
            ground_truth = tensor_to_pil(enhanced_tensor)
            images.append(ground_truth)
            titles.append("Ground Truth")

        # Plot results
        plot_results(
            images, titles, figure_size=(15, 5), save_path=f"test_result_{i + 1}.png"
        )


def test_single_image(model, image_path, device, save_dir="results"):
    """
    Test model on a single image

    Args:
        model: Trained MIRNet model
        image_path (str): Path to input image
        device: Device to run inference on
        save_dir (str): Directory to save results
    """
    print(f"\nProcessing single image: {image_path}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Get image name for saving
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{image_name}_enhanced.png")

    # Perform inference
    original_image, enhanced_image = infer_single_image(
        model, image_path, device, save_path
    )

    # Plot comparison
    plot_results(
        [original_image, enhanced_image],
        ["Original", "MIRNet Enhanced"],
        figure_size=(12, 6),
        save_path=os.path.join(save_dir, f"{image_name}_comparison.png"),
    )

    print(f"Results saved in: {save_dir}")


def evaluate_test_set(model, data_loaders, device):
    """
    Evaluate model on entire test set

    Args:
        model: Trained MIRNet model
        data_loaders: Dictionary containing data loaders
        device: Device to run evaluation on
    """
    print("\nEvaluating on test set...")

    # Criterion for evaluation
    criterion = CharbonnierLoss()

    # Evaluate
    test_metrics = evaluate_model(model, data_loaders["test"], device, criterion)

    # Print results
    print("Test Set Results:")
    print("-" * 30)
    print(f"Loss: {test_metrics['loss']:.6f}")
    print(f"PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"SSIM: {test_metrics['ssim']:.4f}")

    return test_metrics


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="MIRNet Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.MODEL_SAVE_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to single image for inference",
    )
    parser.add_argument(
        "--test_dataset", action="store_true", help="Test on dataset samples"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on entire test set"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of samples for dataset testing",
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(config.RANDOM_SEED)

    # Device setup - fixed for CPU only
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load trained model
    model = load_trained_model(args.model_path, device)
    if model is None:
        return

    # Single image inference
    if args.image_path:
        if os.path.exists(args.image_path):
            test_single_image(model, args.image_path, device, args.save_dir)
        else:
            print(f"Image not found: {args.image_path}")
            return

    # Test on dataset samples
    if args.test_dataset:
        try:
            data_loaders = create_data_loaders(config)
            test_on_dataset(model, data_loaders, device, args.num_samples)
        except Exception as e:
            print(f"Error loading dataset: {e}")

    # Evaluate on test set
    if args.evaluate:
        try:
            data_loaders = create_data_loaders(config)
            evaluate_test_set(model, data_loaders, device)
        except Exception as e:
            print(f"Error during evaluation: {e}")

    # Default behavior if no specific action is requested
    if not any([args.image_path, args.test_dataset, args.evaluate]):
        print("No specific action requested. Testing on dataset samples...")
        try:
            data_loaders = create_data_loaders(config)
            test_on_dataset(model, data_loaders, device, args.num_samples)
        except Exception as e:
            print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()
