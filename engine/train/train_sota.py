# /engine/train/train_sota.py

import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import logging

from engine.model.model_sota import build_sota_model
from engine.train.datagenerator_sota import SOTADataGenerator
from engine.utils.log import setup_logger
from engine.utils.callbacks import RichLoggerCallback

def run_sota_training():
    LOG_DIR = Path("logs")
    logger = setup_logger('Trainer_SOTA', LOG_DIR / 'training_sota.log')
    
    # --- Configuration ---
    INPUT_CSV_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"
    PROCESSED_IMAGE_DIR = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/processed_images_sota"
    MODEL_SAVE_PATH = "pinn_bcd_sota_model.keras"
    
    # EfficientNet is large, so a smaller batch size is necessary
    BATCH_SIZE = 4
    EPOCHS = 15  # Fine-tuning often requires fewer epochs
    VALIDATION_SPLIT = 0.15

    logger.info("--- Starting State-of-the-Art Model Training ---")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        train_df, val_df = train_test_split(df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df['label'])
        
        class_labels = np.unique(train_df['label'])
        weights = compute_class_weight('balanced', classes=class_labels, y=train_df['label'])
        class_weight_dict = dict(zip(class_labels, weights))
        logger.info(f"Calculated class weights: {class_weight_dict}")

        train_generator = SOTADataGenerator(train_df, PROCESSED_IMAGE_DIR, batch_size=BATCH_SIZE, shuffle=True, augment=True)
        val_generator = SOTADataGenerator(val_df, PROCESSED_IMAGE_DIR, batch_size=BATCH_SIZE, shuffle=False)
        
        temp_x, _ = train_generator[0]
        clinical_input_shape = (temp_x['clinical_input'].shape[1],)
        
        model = build_sota_model(clinical_input_shape=clinical_input_shape)
        model.summary(print_fn=lambda x: logger.info(x))
        
        callbacks = [
            RichLoggerCallback(),
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_auc', mode='max'),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=3, mode='max', min_lr=1e-7, verbose=1)
        ]

        logger.info("--- Starting SOTA model.fit() ---")
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            # verbose=0
        )
        
        logger.info("--- SOTA Training Complete! ---")

    except Exception as e:
        logger.exception(f"An unrecoverable SOTA training error occurred: {e}")
    
if __name__ == '__main__':
    run_sota_training()