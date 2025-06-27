# /preprocessing/imageprocessor.py

import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import logging

from engine.utils.patch import PatchExtractor
from engine.utils.log import setup_logger

def process_and_save_images(df: pd.DataFrame, output_base_dir: str, logger: logging.Logger):
    patch_extractor = PatchExtractor(patch_size=128, stride=64, k=5)
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting image processing. Output will be saved to: {output_base_dir}")
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        try:
            image_id_unique = row['unique_id']
            image_output_dir = os.path.join(output_base_dir, image_id_unique)
            Path(image_output_dir).mkdir(exist_ok=True)

            image_path = row['image file path']
            full_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if full_image is None:
                logger.warning(f"Could not read image at {image_path}. Skipping.")
                continue

            top_patches, _ = patch_extractor.extract(full_image)
            global_image = cv2.resize(full_image, (128, 128), interpolation=cv2.INTER_AREA)
            
            global_image_norm = (global_image.astype(np.float32) / 255.0)
            global_image_final = np.expand_dims(global_image_norm, axis=-1)
            top_patches_final = np.expand_dims(top_patches, axis=-1)

            np.save(os.path.join(image_output_dir, 'patches.npy'), top_patches_final)
            np.save(os.path.join(image_output_dir, 'global.npy'), global_image_final)

        except Exception as e:
            logger.exception(f"ERROR processing image for id {row.get('unique_id', 'N/A')}: {e}")
            
    logger.info("Image processing complete.")

def main():
    LOG_DIR = Path("logs")
    logger = setup_logger('ImageProcessor', LOG_DIR / 'image_processing.log')

    INPUT_CSV_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"
    OUTPUT_IMAGE_DIR = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/processed_images"
    
    logger.info("--- Starting Image Processing Pipeline (Final Version) ---")
    
    if not os.path.exists(INPUT_CSV_PATH):
        logger.error(f"Input CSV not found at '{INPUT_CSV_PATH}'.")
        logger.error("Please run the csvprocessor script first.")
        return
        
    df = pd.read_csv(INPUT_CSV_PATH)
    process_and_save_images(df, OUTPUT_IMAGE_DIR, logger)

if __name__ == "__main__":
    main()