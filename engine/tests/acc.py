# /test/evaluate.py

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add project root to Python path for module imports ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import custom modules ---
from utils.patch import PatchExtractor
from utils.log import setup_logger
from train.datagenerator import DataGenerator # To get the fitted preprocessor

# --- Configuration ---
MODEL_PATH = "pinn_bcd_model.keras"
# Use your final, cleaned CSV for evaluation
CSV_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/test.csv"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / 'evaluation.log'
RESULTS_DIR = Path("evaluation_results")

# Model & Processing Parameters (must match training)
PATCH_SIZE = 128
STRIDE = 64
TOP_K = 5
GLOBAL_IMG_SIZE = (128, 128)

# --- Main Evaluation Logic ---

def get_fitted_preprocessor(csv_path: str):
    """ Re-fits the preprocessor on the full dataset to ensure consistency. """
    df = pd.read_csv(csv_path)
    dummy_generator = DataGenerator(df, processed_img_dir="", batch_size=1, shuffle=False)
    return dummy_generator.preprocessor

def evaluate_model(logger: logging.Logger, preprocessor):
    """
    Main function to evaluate the model on the entire dataset specified in the CSV.
    """
    logger.info("--- Starting Full Dataset Evaluation ---")

    # --- 1. Validate paths and load model/data ---
    if not all(os.path.exists(p) for p in [MODEL_PATH, CSV_PATH]):
        logger.error("A required file (model or CSV) was not found. Aborting.")
        return
        
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    df = pd.read_csv(CSV_PATH)
    
    patch_extractor = PatchExtractor(patch_size=PATCH_SIZE, stride=STRIDE, k=TOP_K)

    all_true_labels = []
    all_pred_probs = []

    # --- 2. Iterate through dataset and generate predictions ---
    logger.info(f"Processing {len(df)} images for evaluation...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Model"):
        try:
            # --- Get Image Inputs ---
            full_image = cv2.imread(row['image file path'], cv2.IMREAD_GRAYSCALE)
            if full_image is None: continue

            top_patches_norm, _ = patch_extractor.extract(full_image)
            top_patches_final = np.expand_dims(top_patches_norm, axis=-1)
            
            global_image = cv2.resize(full_image, GLOBAL_IMG_SIZE, interpolation=cv2.INTER_AREA)
            global_image_norm = (global_image.astype(np.float32) / 255.0)
            global_image_final = np.expand_dims(global_image_norm, axis=-1)
            
            # --- Get CSV Input ---
            # Create a single-row DataFrame to be transformed
            input_df = pd.DataFrame([row])
            csv_vector = preprocessor.transform(input_df[preprocessor.feature_names_in_])

            # --- Prepare Final Inputs ---
            inputs = {
                "topk_input": np.expand_dims(top_patches_final, axis=0),
                "global_input": np.expand_dims(global_image_final, axis=0),
                "clinical_input": csv_vector.astype(np.float32)
            }
            
            # --- Predict and Store ---
            pred_prob = model.predict(inputs, verbose=0)[0][0]
            all_pred_probs.append(pred_prob)
            all_true_labels.append(row['label'])

        except Exception as e:
            logger.warning(f"Skipping row for unique_id {row['unique_id']} due to error: {e}")
            continue

    # --- 3. Calculate and Display Metrics ---
    logger.info("Evaluation complete. Calculating final metrics...")
    
    # Convert probabilities to binary classes (0 or 1) using a 0.5 threshold
    all_pred_classes = (np.array(all_pred_probs) > 0.5).astype(int)
    all_true_labels = np.array(all_true_labels)

    # Accuracy and AUC Score
    accuracy = accuracy_score(all_true_labels, all_pred_classes)
    auc = roc_auc_score(all_true_labels, all_pred_probs) # Use probabilities for AUC

    # Classification Report (Precision, Recall, F1-Score)
    class_report_str = classification_report(all_true_labels, all_pred_classes, target_names=['Benign (0)', 'Malignant (1)'])
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_pred_classes)
    tn, fp, fn, tp = cm.ravel()

    # --- Log and Print Comprehensive Report ---
    report = (
        "\n"
        "==================== MODEL EVALUATION REPORT ====================\n"
        f"Total Samples Evaluated: {len(all_true_labels)}\n\n"
        f"--- Key Metrics ---\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"ROC AUC Score: {auc:.4f}\n\n"
        f"--- Confusion Matrix ---\n"
        f"                Predicted Benign | Predicted Malignant\n"
        f"True Benign    :    {tn:^10}    |    {fp:^10}\n"
        f"True Malignant :    {fn:^10}    |    {tp:^10}\n\n"
        f"Breakdown:\n"
        f" - Correctly predicted Benign (True Negatives): {tn}\n"
        f" - Correctly predicted Malignant (True Positives): {tp}\n"
        f" - Incorrectly predicted Malignant (False Positives): {fp}\n"
        f" - Incorrectly predicted Benign (False Negatives): {fn}\n\n"
        f"--- Classification Report ---\n"
        f"{class_report_str}\n"
        "=================================================================\n"
    )
    logger.info(report)
    print(report)
    
    # --- Save Confusion Matrix Plot ---
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plot_path = RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(plot_path)
    logger.info(f"Confusion matrix plot saved to: {plot_path}")

def main():
    Path(LOG_DIR).mkdir(exist_ok=True)
    logger = setup_logger('ModelEvaluator', LOG_FILE)
    
    logger.info("Initializing preprocessor for CSV data...")
    preprocessor = get_fitted_preprocessor(CSV_PATH)
    
    if preprocessor:
        evaluate_model(logger, preprocessor)

if __name__ == "__main__":
    main()