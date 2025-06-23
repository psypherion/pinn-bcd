# lbhist.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "./csv/train.csv" # This CSV should now have resnet50_feat_path and lbp_hist_path
RESNET_FEAT_COL = "resnet50_feat_path" # Column name for ResNet50 features
LBP_FEAT_COL = "lbp_hist_path"       # Column name for LBP features
LABEL_COL = "label_3class"           # Crucial: Use the 3-class label

# Output directory for fused features
OUT_DIR = "./fused"
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD CSV ===
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"[Fusion] CSV file not found at {CSV_PATH}. Please run prerequisite scripts first.")
    exit()

if RESNET_FEAT_COL not in df.columns:
    print(f"[Fusion] ResNet feature path column '{RESNET_FEAT_COL}' not found in {CSV_PATH}.")
    print("         Please run the ResNet50 feature extraction script (reshist.py) first.")
    exit()
if LBP_FEAT_COL not in df.columns:
    print(f"[Fusion] LBP feature path column '{LBP_FEAT_COL}' not found in {CSV_PATH}.")
    print("         Please run the LBP feature generation script (lbp_generator.py) first.")
    exit()
if LABEL_COL not in df.columns:
    print(f"[Fusion] Label column '{LABEL_COL}' not found in {CSV_PATH}.")
    exit()


# Drop the ROI_mask_file_path column if it exists and isn't needed for fusion
# This was in your original lbhist.py, keeping it if it's still relevant.
if "ROI_mask_file_path" in df.columns:
    df = df.drop(columns=["ROI_mask_file_path"])
    print("[Fusion] Dropped 'ROI_mask_file_path' column.")


lbp_vectors = []
resnet_vectors = []
labels_for_y = []
keep_indices = [] # Original indices from the df that had valid features

print(f"[Fusion] Starting feature fusion from '{CSV_PATH}'")

# === Extract valid feature rows ===
for idx, row in tqdm(df.iterrows(), total=len(df), desc="[Fusion] Reading features and labels"):
    resnet_path_val = row[RESNET_FEAT_COL]
    lbp_path_val = row[LBP_FEAT_COL]

    # Check if paths are valid (not NaN and exist)
    if pd.isna(resnet_path_val) or not os.path.exists(str(resnet_path_val)):
        # print(f"[!] Missing or invalid ResNet path for original index {idx}: {resnet_path_val}")
        continue
    if pd.isna(lbp_path_val) or not os.path.exists(str(lbp_path_val)):
        # print(f"[!] Missing or invalid LBP path for original index {idx}: {lbp_path_val}")
        continue
    
    current_label = row[LABEL_COL]
    if pd.isna(current_label):
        # print(f"[!] Missing label for original index {idx}.")
        continue

    try:
        # It's safer to use str() for paths read from pandas
        resnet_feat = np.load(str(resnet_path_val))
        lbp_feat = np.load(str(lbp_path_val))
    except Exception as e:
        print(f"[!] Failed to load feature .npy for original index {idx}: {e}")
        print(f"    ResNet path: {resnet_path_val}, LBP path: {lbp_path_val}")
        continue

    # Sanity check dimensions (optional but good)
    # ResNet50 should be 2048, LBP should be 59
    if resnet_feat.shape[0] != 2048: # Assuming ResNet50 output dim
        print(f"[!] Warning: ResNet feature for index {idx} has shape {resnet_feat.shape}, expected (2048,). Skipping.")
        continue
    if lbp_feat.shape[0] != NUM_LBP_BINS: # NUM_LBP_BINS is 59 from lbp_generator
        print(f"[!] Warning: LBP feature for index {idx} has shape {lbp_feat.shape}, expected ({NUM_LBP_BINS},). Skipping.")
        continue


    resnet_vectors.append(resnet_feat)
    lbp_vectors.append(lbp_feat)
    labels_for_y.append(current_label)
    keep_indices.append(idx) # Store original DataFrame index

# === Final fused arrays ===
if not keep_indices:
    print("[Fusion] No valid samples found after attempting to load features. Exiting.")
    exit()

X_resnet = np.array(resnet_vectors, dtype=np.float32)
X_lbp = np.array(lbp_vectors, dtype=np.float32)
y = np.array(labels_for_y, dtype=np.int64) # Labels for classification should be int

# === Save to disk ===
X_RESNET_OUT_PATH = os.path.join(OUT_DIR, "X_resnet.npy")
X_LBP_OUT_PATH = os.path.join(OUT_DIR, "X_lbp.npy")
Y_OUT_PATH = os.path.join(OUT_DIR, "y.npy")

np.save(X_RESNET_OUT_PATH, X_resnet)
np.save(X_LBP_OUT_PATH, X_lbp)
np.save(Y_OUT_PATH, y)

print(f"[Fusion] Saved fused features:")
print(f"    - ResNet features : {X_RESNET_OUT_PATH} (Shape: {X_resnet.shape})")
print(f"    - LBP features    : {X_LBP_OUT_PATH} (Shape: {X_lbp.shape})")
print(f"    - Labels          : {Y_OUT_PATH} (Shape: {y.shape})")


# === Update train.csv with feature index pointers ===
# Create a new DataFrame containing only the rows that were successfully fused
df_fused = df.loc[keep_indices].reset_index(drop=True)
# Add 'fused_index' which maps rows in this new df_fused to indices in X_resnet, X_lbp, y
df_fused["fused_index"] = np.arange(len(df_fused)) 

# Save this filtered and indexed CSV
FUSED_CSV_PATH = os.path.join(OUT_DIR, "train_fused.csv") # Save as a new CSV
df_fused.to_csv(FUSED_CSV_PATH, index=False)

# Optionally, if you want to overwrite the original ./csv/train.csv:
# df_all_original = pd.read_csv(CSV_PATH) # Reload original to not lose unfused rows
# df_all_original['fused_index'] = pd.NA # Initialize with NA
# df_all_original.loc[keep_indices, 'fused_index'] = np.arange(len(df_fused))
# df_all_original.to_csv(CSV_PATH, index=False)
# print(f"[Fusion] Updated original '{CSV_PATH}' with 'fused_index' (NaN for skipped rows).")


print(f"[Fusion] Created '{FUSED_CSV_PATH}' with {len(df_fused)} successfully fused samples.")
print(f"[Fusion] This new CSV contains only the samples for which features were successfully extracted and fused.")
print(f"[Fusion] The 'fused_index' column in '{FUSED_CSV_PATH}' corresponds to the indices in the .npy files.")
print(f"[Fusion] Original CSV path: {CSV_PATH} (remains unchanged or can be optionally updated).")