# /predict/predict.py

import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from pathlib import Path
import pandas as pd
# --- Add project root to Python path for module imports ---
import sys
# This assumes the script is in /predict, so we go up one level to pinn-bcd/
# And then add the root directory, which is one more level up.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import custom modules ---
from utils.patch import PatchExtractor
from utils.log import setup_logger
from train.datagenerator import DataGenerator # We need this to get the preprocessor

# --- Hardcoded Global Variables & Configuration ---
MODEL_PATH = "pinn_bcd_model.keras"
INPUT_IMAGE_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/1-009.jpg"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / 'prediction.log'

# --- These MUST match your training configuration ---
PATCH_SIZE = 128
STRIDE = 64
TOP_K = 5
GLOBAL_IMG_SIZE = (128, 128)

# --- Optional User Metadata ---
USER_METADATA = {
    # 'breast_density': 3,
    # 'left or right breast': 'LEFT',
    # 'image view': 'CC',
    # 'abnormality type': 'mass'
}
# USER_METADATA = None # Uncomment for prediction with no extra data

# --- THE BIG FIX: A professional way to handle CSV preprocessing for prediction ---
# We will create a dummy DataGenerator instance just to get its fitted preprocessor.
# This ensures that the encoding is EXACTLY the same as during training.
def get_fitted_preprocessor():
    """
    Loads the training data to fit a ColumnTransformer, ensuring
    prediction encoding matches training encoding.
    """
    csv_path = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"
    if not os.path.exists(csv_path):
        print("ERROR: Training CSV not found. Cannot create preprocessor for CSV data.")
        return None
    
    df = pd.read_csv(csv_path)

    dummy_generator = DataGenerator(df, processed_img_dir="", batch_size=1, shuffle=False)
    return dummy_generator.preprocessor

def create_csv_vector_from_user_data(user_metadata, preprocessor):
    """
    Creates the clinical feature vector using the same preprocessor from training.
    """
    # Create a DataFrame with a single row of default/user data
    default_data = {
        'breast_density': 2, # A common default
        'left or right breast': 'LEFT',
        'image view': 'CC',
        'abnormality type': 'mass'
    }
    
    if user_metadata:
        # Update defaults with any provided user data
        default_data.update(user_metadata)

    # Create a pandas DataFrame from the dictionary
    input_df = pd.DataFrame([default_data])
    
    # Use the FITTED preprocessor from the DataGenerator to transform the data
    processed_vector = preprocessor.transform(input_df)
    
    return processed_vector.astype(np.float32)

def predict_single_image(logger: logging.Logger, preprocessor):
    """ Main function to run a single prediction. """
    logger.info("--- Starting Single Image Prediction (Corrected) ---")
    
    if not all(os.path.exists(p) for p in [MODEL_PATH, INPUT_IMAGE_PATH]):
        logger.error(f"FATAL: A required file was not found. Check paths. Cannot proceed.")
        return

    logger.info(f"Loading model: {MODEL_PATH}")
    logger.info(f"Loading input image: {INPUT_IMAGE_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        patch_extractor = PatchExtractor(patch_size=PATCH_SIZE, stride=STRIDE, k=TOP_K)

        full_image = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if full_image is None:
            logger.error("Failed to read the input image file."); return

        top_patches_norm, _ = patch_extractor.extract(full_image)
        top_patches_final = np.expand_dims(top_patches_norm, axis=-1)

        global_image = cv2.resize(full_image, GLOBAL_IMG_SIZE, interpolation=cv2.INTER_AREA)
        global_image_norm = (global_image.astype(np.float32) / 255.0)
        global_image_final = np.expand_dims(global_image_norm, axis=-1)

        csv_vector = create_csv_vector_from_user_data(USER_METADATA, preprocessor)
        
        # --- THE FIX IS HERE: Use the correct dictionary keys ---
        inputs = {
            "topk_input": np.expand_dims(top_patches_final, axis=0),
            "global_input": np.expand_dims(global_image_final, axis=0),
            "clinical_input": csv_vector # Already has a batch dimension of 1
        }
        
        logger.info("Inputs prepared correctly. Making prediction...")
        prediction_prob = model.predict(inputs)[0][0]

        diagnosis = "POSITIVE for Malignancy (High Concern)" if prediction_prob > 0.5 else "NEGATIVE for Malignancy (Low Concern)"
        
        result_string = (
            "\n"
            "------------------- DIAGNOSIS -------------------\n"
            f"Model Confidence (Probability of Malignancy): {prediction_prob:.4f}\n"
            f"Final Result: {diagnosis}\n"
            "-------------------------------------------------\n"
            "Disclaimer: This is an AI-powered educational tool and not a substitute for professional medical advice.\n"
        )
        logger.info(result_string); print(result_string)

    except Exception as e:
        logger.exception(f"An unexpected error occurred during prediction: {e}")

def main():
    Path(LOG_DIR).mkdir(exist_ok=True)
    logger = setup_logger('PredictionScript', LOG_FILE)
    
    logger.info("Initializing preprocessor for CSV data...")
    # This is a critical step to prevent data leakage and ensure consistency
    preprocessor = get_fitted_preprocessor()
    
    if preprocessor:
        predict_single_image(logger, preprocessor)

if __name__ == "__main__":
    main()