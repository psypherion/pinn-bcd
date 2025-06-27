# /preprocessing/imageprocessor_sota.py

import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import logging

# Add project root to path for module imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.utils.patch import PatchExtractor
from engine.utils.log import setup_logger

# SOTA Configuration: Input size for EfficientNetB0
SOTA_IMG_SIZE = 224

def process_and_save_images_sota(df: pd.DataFrame, output_base_dir: str, logger: logging.Logger):
    # Use larger patches initially to have more detail before resizing
    patch_extractor = PatchExtractor(patch_size=256, stride=128, k=5)
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting SOTA image processing. Output size: {SOTA_IMG_SIZE}x{SOTA_IMG_SIZE}")
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing SOTA Images"):
        try:
            image_output_dir = Path(output_base_dir) / row['unique_id']
            image_output_dir.mkdir(exist_ok=True)

            full_image = cv2.imread(row['image file path'], cv2.IMREAD_GRAYSCALE)
            if full_image is None:
                logger.warning(f"Could not read image: {row['image file path']}. Skipping.")
                continue

            # --- Extract Patches ---
            # Patches are still normalized floats (0-1) from the extractor
            top_patches_norm, _ = patch_extractor.extract(full_image)
            
            # --- Resize and Save Patches ---
            resized_patches = []
            for patch in top_patches_norm:
                resized_patch = cv2.resize(patch, (SOTA_IMG_SIZE, SOTA_IMG_SIZE), interpolation=cv2.INTER_AREA)
                resized_patches.append(resized_patch)
            
            # Add the channel dimension for saving
            patches_final = np.expand_dims(np.array(resized_patches), axis=-1)
            np.save(image_output_dir / 'patches_sota.npy', patches_final)

            # --- Resize and Save Global Image ---
            global_image = cv2.resize(full_image, (SOTA_IMG_SIZE, SOTA_IMG_SIZE), interpolation=cv2.INTER_AREA)
            global_image_norm = (global_image.astype(np.float32) / 255.0)
            global_image_final = np.expand_dims(global_image_norm, axis=-1)
            np.save(image_output_dir / 'global_sota.npy', global_image_final)

        except Exception as e:
            logger.exception(f"ERROR processing SOTA image for id {row.get('unique_id', 'N/A')}: {e}")
            
    logger.info("SOTA image processing complete.")

def main():
    LOG_DIR = Path("logs")
    logger = setup_logger('ImageProcessor_SOTA', LOG_DIR / 'image_processing_sota.log')

    INPUT_CSV_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"
    OUTPUT_IMAGE_DIR_SOTA = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/processed_images_sota"
    
    logger.info("--- Starting SOTA Image Processing Pipeline ---")
    
    df = pd.read_csv(INPUT_CSV_PATH)
    process_and_save_images_sota(df, OUTPUT_IMAGE_DIR_SOTA, logger)

if __name__ == "__main__":
    main()