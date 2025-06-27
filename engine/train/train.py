# /engine/train/train.py

import pandas as pd
import os
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Import our custom modules
from engine.model.model import build_pinn_bcd_model
from engine.train.datagenerator import DataGenerator
from engine.utils.log import setup_logger

def setup_cpu_training():
    """
    Configures TensorFlow to explicitly use the CPU.
    This should be called at the very beginning of the script.
    """
    logger = logging.getLogger('Trainer')
    logger.info("--- Configuring for CPU-only training ---")
    
    # Hide GPUs from TensorFlow. This forces it to use the CPU.
    tf.config.set_visible_devices([], 'GPU')
    
    # You can also set intra/inter op parallelism threads for more control,
    # but hiding the GPU is the most direct method.
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 1)
    
    # Verify that no GPUs are visible
    visible_devices = tf.config.get_visible_devices()
    gpu_devices = [device for device in visible_devices if device.device_type == 'GPU']
    if not gpu_devices:
        logger.info("✅ TensorFlow is set to use CPU only. No GPUs are visible.")
    else:
        logger.warning("Could not hide GPUs. TensorFlow may still attempt to use them.")

def run_training():
    """Main function to configure and run the training process."""
    
    # Setup logging first
    LOG_DIR = Path("logs")
    logger = setup_logger('Trainer', LOG_DIR / 'training.log')
    
    # --- STEP 1: Configure Environment for CPU ---
    setup_cpu_training()
    
    # --- Configuration ---
    INPUT_CSV_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"
    PROCESSED_IMAGE_DIR = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/processed_images"
    MODEL_SAVE_PATH = "pinn_bcd_model.keras"
    
    BATCH_SIZE = 16
    EPOCHS = 30
    VALIDATION_SPLIT = 0.2

    logger.info("--- Starting Model Training (CPU Optimized) ---")

    try:
        if not os.path.exists(INPUT_CSV_PATH):
            logger.error(f"Input CSV not found at '{INPUT_CSV_PATH}'. Aborting."); return
        df = pd.read_csv(INPUT_CSV_PATH)

        train_df, val_df = train_test_split(df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df['label'])
        logger.info(f"Data split into {len(train_df)} training and {len(val_df)} validation samples.")
        
        train_generator = DataGenerator(train_df, PROCESSED_IMAGE_DIR, batch_size=BATCH_SIZE, shuffle=True, augment=True)
        val_generator = DataGenerator(val_df, PROCESSED_IMAGE_DIR, batch_size=BATCH_SIZE, shuffle=False, augment=False)
        
        temp_x, _ = train_generator[0]
        if not temp_x:
            logger.error("Data generator returned empty first batch. Aborting."); return
            
        clinical_input_shape = (temp_x['clinical_input'].shape[1],)
        logger.info(f"Determined clinical input shape: {clinical_input_shape}")
        
        # Build model within the CPU context
        with tf.device('/CPU:0'):
            model = build_pinn_bcd_model(clinical_input_shape=clinical_input_shape)
        logger.info("Model built successfully on CPU device.")
        
        stringlist = []; model.summary(print_fn=lambda x: stringlist.append(x)); model_summary = "\n".join(stringlist)
        logger.info(f"Model Summary:\n{model_summary}")
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_auc', mode='max', verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR / 'tensorboard_logs'), histogram_freq=1)
        ]

        logger.info(f"--- Starting model.fit() for {EPOCHS} epochs ---")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
        )
        
        logger.info("--- Training Complete! ---")
        logger.info(f"Best model saved to '{MODEL_SAVE_PATH}'")

    except Exception as e:
        logger.exception(f"An unrecoverable error occurred during training: {e}")
    
if __name__ == '__main__':
    run_training()