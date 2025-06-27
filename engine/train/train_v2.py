# /engine/train/train_v2.py

import pandas as pd
import os
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import logging

# --- Import custom modules ---
from engine.model.model_v2 import build_pinn_bcd_model_v2
from engine.train.datagenerator import DataGenerator
from engine.utils.log import setup_logger
from engine.utils.callbacks import RichLoggerCallback # <-- IMPORT OUR NEW CALLBACK

def run_advanced_training():
    """Main function for the V2 training process with all regularization techniques."""
    
    LOG_DIR = Path("logs")
    logger = setup_logger('Trainer_V2', LOG_DIR / 'training_v2.log')
    
    # --- Configuration ---
    INPUT_CSV_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"
    PROCESSED_IMAGE_DIR = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/processed_images"
    MODEL_SAVE_PATH = "pinn_bcd_model_v2.keras"
    
    BATCH_SIZE = 16
    EPOCHS = 69
    VALIDATION_SPLIT = 0.25

    logger.info("--- Starting ADVANCED Model Training (V2 - All Strategies) ---")

    try:
        # ... (The data loading, splitting, and class weight calculation part remains the same)
        df = pd.read_csv(INPUT_CSV_PATH)
        train_df, val_df = train_test_split(df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df['label'])
        
        class_labels = np.unique(train_df['label'])
        weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=train_df['label'])
        class_weight_dict = dict(zip(class_labels, weights))
        logger.info(f"Calculated class weights: {class_weight_dict}")

        train_generator = DataGenerator(train_df, PROCESSED_IMAGE_DIR, batch_size=BATCH_SIZE, shuffle=True, augment=True)
        val_generator = DataGenerator(val_df, PROCESSED_IMAGE_DIR, batch_size=BATCH_SIZE, shuffle=False, augment=False)
        
        temp_x, _ = train_generator[0]
        clinical_input_shape = (temp_x['clinical_input'].shape[1],)
        
        model = build_pinn_bcd_model_v2(clinical_input_shape=clinical_input_shape)
        model.summary(print_fn=lambda x: logger.info(x))
        
        # --- Updated Callbacks List ---
        callbacks = [
            # Our custom logger comes first
            RichLoggerCallback(), 
            
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_SAVE_PATH, save_best_only=True, monitor='val_auc', mode='max', verbose=0 # Set verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.2, patience=4, mode='max', min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(LOG_DIR / 'tensorboard_logs_v2'), histogram_freq=1
            )
        ]

        logger.info("--- Starting model.fit() with custom logging ---")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight_dict,
        )
        
        logger.info("--- V2 Training Complete! ---")
        logger.info(f"Best model saved to '{MODEL_SAVE_PATH}'")

    except Exception as e:
        logger.exception(f"An unrecoverable error occurred during training: {e}")
    
if __name__ == '__main__':
    run_advanced_training()