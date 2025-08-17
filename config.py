"""
Configuration file for MIRNet training and model parameters
"""

import os

# Model Parameters
RRB_COUNT = 3  # Number of Recursive Residual Blocks
MSRB_COUNT = 2  # Number of Multi-Scale Residual Blocks per RRB
CHANNELS = 64  # Number of channels for feature processing

# Training Parameters
IMAGE_SIZE = 128
BATCH_SIZE = 4
MAX_TRAIN_IMAGES = 300
EPOCHS = 50
LEARNING_RATE = 1e-4

# Data Paths
DATA_ROOT = "/Applications/ML projects/Success/Low Light Image Enhancement/lol_dataset"
TRAIN_LOW_PATH = os.path.join(DATA_ROOT, "our485/low/*")
TRAIN_HIGH_PATH = os.path.join(DATA_ROOT, "our485/high/*")
TEST_LOW_PATH = os.path.join(DATA_ROOT, "eval15/low/*")
TEST_HIGH_PATH = os.path.join(DATA_ROOT, "eval15/high/*")

# Training Settings
RANDOM_SEED = 10
DEVICE = "cpu"

# Scheduler Parameters
LR_PATIENCE = 5  # Patience for learning rate reduction
LR_FACTOR = 0.5  # Factor for learning rate reduction
MIN_LR_DELTA = 1e-7  # Minimum delta for improvement

# Output
MODEL_SAVE_PATH = "model.pth"
CHECKPOINT_DIR = "checkpoints"
