# Low-Light-Image-Enhancement

# MIRNet PyTorch Implementation

A PyTorch implementation of **MIRNet (Multi-scale Image Restoration Network)** for low-light image enhancement. This implementation converts the original TensorFlow/Keras code to PyTorch while maintaining the same architecture and approach.

## Project Structure

```
mirnet_pytorch/
├── config.py          # Configuration parameters
├── blocks.py           # Neural network building blocks
├── model.py            # Main MIRNet model definition
├── data.py             # Data loading and preprocessing
├── utils.py            # Utility functions (loss, metrics, etc.)
├── train.py            # Training script
├── inference.py        # Inference and testing script
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Model Architecture

### Key Components

1. **Selective Kernel Feature Fusion (SKFF)**: Intelligently combines multi-scale features using attention mechanisms
2. **Dual Attention Unit (DAU)**: Combines channel and spatial attention for enhanced feature representation
3. **Multi-Scale Residual Block (MSRB)**: Processes images at multiple scales simultaneously
4. **Recursive Residual Block (RRB)**: Stacks multiple MSRBs for deep feature extraction

### Architecture Overview

```
Input Image → Conv → RRB₁ → RRB₂ → RRB₃ → Conv → + → Enhanced Image
     ↑                                              ↗
     └─────────────── Residual Connection ─────────┘
```

Each RRB contains multiple MSRBs, and each MSRB processes features at 3 different scales using DAU and SKFF.

## Requirements

```bash
pip install torch torchvision matplotlib pillow numpy tqdm
```

## Dataset

The code expects the LOL (Low-Light) dataset with the following structure:

```
lol_dataset/
├── our485/
│   ├── low/     # Training low-light images
│   └── high/    # Training enhanced images
└── eval15/
    ├── low/     # Test low-light images
    └── high/    # Test enhanced images
```

Update the data paths in `config.py` to match your dataset location.

## Configuration

Modify `config.py` to adjust training parameters:

```python
# Model Parameters
RRB_COUNT = 3           # Number of Recursive Residual Blocks
MSRB_COUNT = 2          # Number of Multi-Scale Residual Blocks per RRB
CHANNELS = 64           # Number of channels

# Training Parameters
IMAGE_SIZE = 128        # Training image size
BATCH_SIZE = 4          # Batch size
EPOCHS = 50             # Number of epochs
LEARNING_RATE = 1e-4    # Learning rate

# Data Paths
DATA_ROOT = "/path/to/lol_dataset"
```

## Usage

### Training

Start training with default parameters:

```bash
python train.py
```

The training script will:
- Load and preprocess the dataset
- Create the MIRNet model
- Train for the specified number of epochs
- Save checkpoints and the best model
- Evaluate on the test set

### Inference

#### Test on dataset samples
```bash
python inference.py --test_dataset --num_samples 6
```

#### Test on a single image
```bash
python inference.py --image_path /path/to/image.jpg --save_dir results/
```

#### Evaluate on entire test set
```bash
python inference.py --evaluate
```

#### Load specific model
```bash
python inference.py --model_path /path/to/model.pth --test_dataset
```

### Model Testing

Test the model structure:

```bash
python model.py
```

Test data loading:

```bash
python data.py
```

Test utility functions:

```bash
python utils.py
```

## Model Performance

The model is trained using:
- **Loss Function**: Charbonnier Loss (more robust than MSE)
- **Metric**: PSNR (Peak Signal-to-Noise Ratio)
- **Optimizer**: Adam with learning rate 1e-4
- **Scheduler**: ReduceLROnPlateau (monitors validation PSNR)

## File Descriptions

### `config.py`
Contains all configuration parameters for easy modification.

### `blocks.py`
Implements all neural network building blocks:
- `SelectiveKernelFeatureFusion`
- `ChannelAttentionBlock` & `SpatialAttentionBlock`
- `DualAttentionUnitBlock`
- `DownSamplingBlock` & `UpSamplingBlock`
- `MultiScaleResidualBlock`
- `RecursiveResidualBlock`

### `model.py`
Main MIRNet model implementation that combines all blocks.

### `data.py`
Data loading and preprocessing:
- `LOLDataset` for training/validation
- `TestDataset` for inference
- Data augmentation and random cropping

### `utils.py`
Utility functions:
- `CharbonnierLoss` - Custom loss function
- `peak_signal_noise_ratio` - PSNR calculation
- Model saving/loading utilities
- Inference and evaluation functions

### `train.py`
Complete training pipeline with:
- Training and validation loops
- Progress tracking
- Checkpoint management
- Learning rate scheduling

### `inference.py`
Testing and inference script with multiple modes:
- Single image enhancement
- Dataset sample testing
- Full test set evaluation

## Tips for Training

1. **GPU Memory**: Reduce `BATCH_SIZE` if you encounter GPU memory issues
2. **Training Time**: Each epoch takes ~2-3 minutes on a modern GPU
3. **Convergence**: The model typically converges within 30-40 epochs
4. **Best Results**: Monitor validation PSNR; best models usually achieve 20+ dB PSNR

## Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Ensure you're using GPU and set `num_workers > 0`
3. **Dataset Not Found**: Check paths in `config.py`
4. **Model Loading Error**: Ensure model architecture matches saved weights

## Citation

If you use this implementation, please cite the original MIRNet paper:

```bibtex
@misc{zamir2020learningenrichedfeaturesreal,
      title={Learning Enriched Features for Real Image Restoration and Enhancement}, 
      author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
      year={2020},
      eprint={2003.06792},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2003.06792}, 
}
```
