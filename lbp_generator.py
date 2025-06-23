# lbp_generator.py
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "./csv/train.csv"
CROPPED_IMAGE_COL = "cropped_image_file_path" # Column in CSV for cropped ROI images
OUTPUT_DIR = "./features/lbp_hist"
FEATURE_COL_NAME = "lbp_hist_path" # New column name for CSV

# LBP parameters
P = 24  # Number of circularly symmetric neighbour set points (commonly 8, 16, 24)
R = 3   # Radius of circle (commonly 1, 2, 3)
# For P=24, R=3 with 'uniform' method, we get P * (P - 1) + 3 = 24 * 23 + 3 = 552 + 3 = 555 bins.
# The prompt specified 59-dim LBP. This is achieved with P=8, R=1, method='uniform' -> 8*(8-1)+3 = 59
# Let's adjust to match the 59-dim requirement.
P_LBP = 8
R_LBP = 1
METHOD_LBP = 'uniform'
NUM_LBP_BINS = P_LBP * (P_LBP - 1) + 3 # Should be 59 for P=8, R=1, 'uniform'

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[LBP] Output directory: {OUTPUT_DIR}")
print(f"[LBP] Parameters: P={P_LBP}, R={R_LBP}, Method='{METHOD_LBP}', Expected Bins={NUM_LBP_BINS}")


def compute_lbp_histogram(image_path, P, R, method):
    try:
        # Read image in grayscale using OpenCV
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            # print(f"[!] Failed to load image (or image is empty): {image_path}")
            return None
    except Exception as e:
        # print(f"[!] Error loading image {image_path} with OpenCV: {e}")
        return None

    try:
        lbp = local_binary_pattern(image, P, R, method=method)
        
        # Calculate histogram of LBP
        # For 'uniform' LBP, the number of bins is P * (P - 1) + 3
        # For other methods, it can be 2**P
        n_bins = int(lbp.max() + 1)
        if method == 'uniform':
             # For uniform LBP, max value is P*(P-1)+2, so P*(P-1)+3 bins
            expected_bins = P * (P - 1) + 3
            if n_bins > expected_bins : # this can happen if non-uniform patterns are present and not mapped
                # this scenario ideally shouldn't happen if skimage handles uniform correctly
                # but as a fallback, ensure hist is of correct size.
                # However, it's better to ensure LBP values are within range.
                # For uniform, values are 0 to P*(P-1)+2
                lbp = np.clip(lbp, 0, expected_bins -1)
                n_bins = expected_bins

        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize histogram (sum to 1)
        hist = hist.astype("float")
        if np.sum(hist) > 0:
            hist /= (np.sum(hist))
        else: # Handle case of all-zero histogram (e.g., completely black image)
            hist = np.zeros(n_bins, dtype=float)

        if len(hist) != NUM_LBP_BINS:
             print(f"[!] Warning: LBP for {image_path} resulted in {len(hist)} bins, expected {NUM_LBP_BINS}. Padding/truncating.")
             # Pad or truncate if necessary, though ideally this shouldn't happen with fixed P,R,method
             if len(hist) < NUM_LBP_BINS:
                 hist = np.pad(hist, (0, NUM_LBP_BINS - len(hist)), 'constant')
             else:
                 hist = hist[:NUM_LBP_BINS]
        
        return hist.astype(np.float32) # Ensure float32 for consistency
    
    except Exception as e:
        # print(f"[!] Error computing LBP for {image_path}: {e}")
        return None

def process_all_lbp(csv_path, cropped_image_col_name, feature_col_to_add):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] CSV file not found at {csv_path}. Please run csvprocessor.py first.")
        return

    if cropped_image_col_name not in df.columns:
        print(f"[!] Cropped image column '{cropped_image_col_name}' not found in {csv_path}.")
        return

    lbp_feature_paths = []
    processed_count = 0
    failed_count = 0

    print(f"[LBP] Starting LBP feature extraction for {len(df)} cropped images...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting LBP features"):
        cropped_image_path = row[cropped_image_col_name]

        if pd.isna(cropped_image_path) or not os.path.exists(str(cropped_image_path)):
            # print(f"[!] Invalid or missing cropped image path for row {index}: {cropped_image_path}")
            lbp_feature_paths.append(np.nan)
            failed_count +=1
            continue

        lbp_hist = compute_lbp_histogram(str(cropped_image_path), P_LBP, R_LBP, METHOD_LBP)

        if lbp_hist is not None:
            # Create a unique filename based on the original cropped image filename
            base_fname = os.path.basename(str(cropped_image_path))
            fname_no_ext = os.path.splitext(base_fname)[0]
            # Ensure unique name if cropped and full images have same base names
            # Adding _lbp to distinguish if needed, though different dir should suffice
            out_filename = f"{fname_no_ext}_lbp.npy" 
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            
            try:
                np.save(out_path, lbp_hist)
                lbp_feature_paths.append(out_path)
                processed_count +=1
            except Exception as e:
                # print(f"[!] Failed to save LBP .npy for {cropped_image_path}: {e}")
                lbp_feature_paths.append(np.nan)
                failed_count +=1
        else:
            lbp_feature_paths.append(np.nan)
            failed_count +=1

    df[feature_col_to_add] = lbp_feature_paths
    
    backup_path = csv_path.replace(".csv", ".lbp.backup.csv")
    try:
        df_existing_backup = pd.read_csv(backup_path) # if it exists, use it
        df_existing_backup[feature_col_to_add] = df[feature_col_to_add] # update only this col
        df_existing_backup.to_csv(backup_path, index=False)
    except FileNotFoundError:
        df.to_csv(backup_path, index=False) # if not, create new backup
    print(f"[LBP] Backup of CSV with new column saved to {backup_path}")
    
    df.to_csv(csv_path, index=False) # Overwrite original CSV
    print(f"[LBP] Successfully processed {processed_count} images for LBP.")
    print(f"[LBP] Failed to process {failed_count} images for LBP.")
    print(f"[LBP] Updated CSV '{csv_path}' with column '{feature_col_to_add}'.")


if __name__ == "__main__":
    process_all_lbp(CSV_PATH, CROPPED_IMAGE_COL, FEATURE_COL_NAME)
    print("[LBP] LBP feature generation complete.")