
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Configuration
CSV_PATH = "./csv/train.csv"  # The CSV from csvprocessor.py
IMAGE_COL = "image_file_path" # Column with full image paths
# If you have train/val/test splits defined in train.csv from csvprocessor.py, use them.
# Otherwise, we'll use a portion of the data or all of it for stats.
# For simplicity, let's assume we use all available images for stats calculation,
# but ideally, you'd use ONLY your defined training split.
# If your 'train.csv' has 'split_train' column (1 for train, 0 otherwise):
USE_TRAIN_SPLIT_COL = True # Set to False if no such column or you want to use all
TRAIN_SPLIT_COL_NAME = 'split_train' # From csvprocessor.py

TARGET_SIZE = (224, 224) # Size images will be resized to (consistent with ResNet)
BATCH_SIZE = 64         # Batch size for calculation

def calculate_mean_std(csv_path, image_col, target_size, batch_size, use_train_split, train_split_col_name):
    print(f"Loading CSV from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV not found at {csv_path}")
        return None, None

    if image_col not in df.columns:
        print(f"Error: Image column '{image_col}' not found in CSV.")
        return None, None

    image_paths = []
    if use_train_split and train_split_col_name in df.columns:
        train_df = df[df[train_split_col_name] == 1]
        if len(train_df) == 0:
            print(f"Warning: No samples found for training split ('{train_split_col_name}' == 1). Using all images.")
            image_paths = df[image_col].dropna().tolist()
        else:
            print(f"Using {len(train_df)} samples from the training split for statistics.")
            image_paths = train_df[image_col].dropna().tolist()
    else:
        if use_train_split: # but column was missing
            print(f"Warning: Train split column '{train_split_col_name}' not found. Using all images for statistics.")
        image_paths = df[image_col].dropna().tolist()
    
    if not image_paths:
        print("Error: No valid image paths found to calculate statistics.")
        return None, None
        
    print(f"Calculating statistics for {len(image_paths)} images.")

    # Simple dataset to load images
    class TempImageDataset(Dataset):
        def __init__(self, file_paths, transform):
            self.file_paths = [fp for fp in file_paths if os.path.exists(str(fp))]
            if len(self.file_paths) != len(file_paths):
                print(f"Warning: Dropped {len(file_paths) - len(self.file_paths)} non-existent image paths.")
            self.transform = transform

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            img_path = self.file_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB') # Convert to RGB for ResNet
                if self.transform:
                    image = self.transform(image)
                return image
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                # print(f"Warning: Skipping image {img_path} due to error: {e}")
                # Return a dummy tensor or handle appropriately; for stats, better to skip
                return torch.zeros((3, target_size[0], target_size[1])) # Dummy tensor


    # Transformation: Resize and convert to tensor
    # NO NORMALIZATION HERE - we are calculating it!
    transform_for_stats = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor() # Scales to [0, 1]
    ])

    dataset = TempImageDataset(image_paths, transform_for_stats)
    # Filter out dummy tensors if any image failed to load properly
    # This part is a bit tricky if getitem returns dummy, DataLoader might still use it.
    # A cleaner way is to ensure TempImageDataset only contains valid image paths.
    # The check `os.path.exists(str(fp))` in TempImageDataset constructor helps.
    
    if len(dataset) == 0:
        print("Error: Dataset is empty after filtering paths. Cannot compute stats.")
        return None, None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.0
    
    pbar = tqdm(loader, desc="Calculating Mean/Std")
    for data_batch in pbar:
        # data_batch is expected to be a batch of image tensors
        if data_batch is None or data_batch.nelement() == 0: # Check for empty or None batch
            continue

        batch_samples = data_batch.size(0) # batch size (the last batch can be smaller)
        data_batch = data_batch.view(batch_samples, data_batch.size(1), -1) # Flatten spatial dims: B, C, H*W
        
        current_mean = data_batch.mean(2).sum(0) # Sum of means per channel over the batch
        current_std = data_batch.std(2).sum(0)   # Sum of stds per channel over the batch
        
        mean += current_mean
        std += current_std
        nb_samples += batch_samples

    if nb_samples == 0:
        print("Error: No samples processed. Cannot compute mean/std.")
        return None, None
        
    mean /= nb_samples
    std /= nb_samples # This calculates mean of stds, not pooled std. For pooled std, a more complex formula is needed.
                     # However, for normalization, mean of per-image stds is often used and fine.
                     # For true pooled std: E[X^2] - (E[X])^2.
                     # We'll stick to this simpler version for now. If you need precise pooled std:
                     # you'd sum E[X] and E[X^2] across batches.

    print(f"\nCalculated over {int(nb_samples)} samples:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")
    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    # Make sure your CSV_PATH is correct and contains the IMAGE_COL
    # and optionally TRAIN_SPLIT_COL_NAME if USE_TRAIN_SPLIT_COL is True.
    
    # If your 'train.csv' (from csvprocessor.py) has a 'split_train' column:
    # Set USE_TRAIN_SPLIT_COL = True
    
    # If 'train.csv' does NOT have 'split_train', or you want to use all images:
    # Set USE_TRAIN_SPLIT_COL = False
    
    dataset_mean, dataset_std = calculate_mean_std(
        CSV_PATH, IMAGE_COL, TARGET_SIZE, BATCH_SIZE, 
        USE_TRAIN_SPLIT_COL, TRAIN_SPLIT_COL_NAME
    )

    if dataset_mean and dataset_std:
        print("\n--- These are your dataset-specific normalization statistics ---")
        print(f"DATASET_MEAN = {dataset_mean}")
        print(f"DATASET_STD = {dataset_std}")
        print("Use these values in your dataset.py transformations.")
    else:
        print("\nFailed to calculate dataset statistics.")