#!/bin/bash

# ==============================================================================
# PINN-BCD: End-to-End Data Processing and Training Pipeline
# ==============================================================================
# This script automates the three main stages of the project:
# 1. CSV Processing: Cleans and prepares the final master CSV.
# 2. Image Processing: Generates patches and global images from the master CSV.
# 3. Model Training: Trains the tri-modal network on the processed data.
#
# Usage:
#   ./run_pipeline.sh
# or to force reprocessing of images:
#   ./run_pipeline.sh --force-reprocess
# ==============================================================================

# --- Configuration ---
# Activate Python virtual environment
source venv/bin/activate

# Define log file for this script
LOG_DIR="logs"
PIPELINE_LOG_FILE="${LOG_DIR}/pipeline_execution.log"

# --- Helper Functions ---
# A function for logging with timestamps to both console and log file
log() {
    message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - PIPELINE - INFO - ${message}" | tee -a "${PIPELINE_LOG_FILE}"
}

# A function for logging errors
log_error() {
    message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - PIPELINE - ERROR - ${message}" | tee -a "${PIPELINE_LOG_FILE}"
}

# Function to print section headers
print_header() {
    title="$1"
    log "=============================================================================="
    log "${title}"
    log "=============================================================================="
}

# --- Script Execution ---

# Ensure log directory exists
mkdir -p "${LOG_DIR}"
# Clear previous pipeline log
> "${PIPELINE_LOG_FILE}"

print_header "Starting PINN-BCD End-to-End Pipeline"

# --- Stage 1: CSV Processing ---
print_header "Stage 1: Processing CSV Files"
log "Running csvprocessor.py to generate the master 'final_training_set.csv'..."

python -m engine.preprocessing.csvprocessor
# Check the exit code of the last command
if [ $? -ne 0 ]; then
    log_error "CSV processing failed. See logs/csv_processing.log for details. Aborting pipeline."
    exit 1
fi
log "✅ CSV processing completed successfully."
log "Master CSV created at kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/final_training_set.csv"


# --- Stage 2: Image Processing ---
print_header "Stage 2: Processing Image Data"
PROCESSED_IMAGE_DIR="/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/processed_images" # Path to check

# Handle forced reprocessing
if [[ "$1" == "--force-reprocess" ]]; then
    log "Force reprocessing requested. Deleting existing processed images directory..."
    if [ -d "${PROCESSED_IMAGE_DIR}" ]; then
        rm -rf "${PROCESSED_IMAGE_DIR}"
        log "Deleted ${PROCESSED_IMAGE_DIR}"
    else
        log "No existing directory to delete."
    fi
fi

# Check if the directory is empty or doesn't exist
if [ ! -d "${PROCESSED_IMAGE_DIR}" ] || [ -z "$(ls -A ${PROCESSED_IMAGE_DIR})" ]; then
    log "Processed images directory is empty or does not exist. Running imageprocessor.py..."
    python -m engine.preprocessing.imageprocessor
    if [ $? -ne 0 ]; then
        log_error "Image processing failed. See logs/image_processing.log for details. Aborting pipeline."
        exit 1
    fi
    log "✅ Image processing completed successfully."
else
    log "✅ Processed images directory already contains data. Skipping image processing."
    log "To force reprocessing, run with: ./run_pipeline.sh --force-reprocess"
fi


# --- Stage 3: Model Training ---
print_header "Stage 3: Training the Tri-Modal Model"
log "Running train.py..."
log "This may take a long time. Monitor logs/training.log and TensorBoard."
log "To view progress with TensorBoard, run in a separate terminal:"
log "  tensorboard --logdir logs/tensorboard_logs"

python -m engine.train.train
if [ $? -ne 0 ]; then
    log_error "Model training failed. See logs/training.log for details. Aborting pipeline."
    exit 1
fi
log "✅ Model training completed successfully."
log "Best model saved to pinn_bcd_model.keras"


# --- Pipeline Finish ---
print_header "🎉 PINN-BCD Pipeline Finished Successfully! 🎉"

# Deactivate virtual environment
deactivate