# reshist.py
import os
import cv2 # Using OpenCV for consistency if LBP generator also uses it
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError # PIL is also good, torchvision often uses it

# === CONFIG ===
CSV_PATH = "./csv/train.csv"
IMAGE_COL = "image_file_path" # Column in CSV for full images
OUTPUT_DIR = "./features/resnet50" # Changed directory name
FEATURE_COL_NAME = "resnet50_feat_path" # New column name for CSV
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[ResNet50] Using device: {DEVICE}")
print(f"[ResNet50] Output directory: {OUTPUT_DIR}")

# === Load ResNet50 (frozen) ===
model = models.resnet50(pretrained=True) # Changed to resnet50
model.fc = torch.nn.Identity()  # Remove final classification layer, output is 2048-dim
model.eval()
model.to(DEVICE)
print("[ResNet50] ResNet50 model loaded.")

# === Image preprocessing ===
# Standard ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_resnet_features(image_path):
    try:
        # Using PIL to open image as torchvision transforms expect PIL Image
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        # print(f"[!] File not found: {image_path}")
        return None
    except UnidentifiedImageError:
        # print(f"[!] Cannot identify image file (corrupted or not an image): {image_path}")
        return None
    except Exception as e:
        # print(f"[!] Error opening image {image_path}: {e}")
        return None

    try:
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model(img_tensor).cpu().numpy().flatten() # Shape (2048,)
        return features
    except Exception as e:
        # print(f"[!] Error during feature extraction for {image_path}: {e}")
        return None

def process_all_resnet(csv_path, image_col_name, feature_col_to_add):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] CSV file not found at {csv_path}. Please run csvprocessor.py first.")
        return

    if image_col_name not in df.columns:
        print(f"[!] Image column '{image_col_name}' not found in {csv_path}.")
        return

    feature_paths_list = []
    processed_count = 0
    failed_count = 0

    print(f"[ResNet50] Starting ResNet50 feature extraction for {len(df)} images...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ResNet50 features"):
        image_path = row[image_col_name]

        if pd.isna(image_path) or not os.path.exists(str(image_path)):
            # print(f"[!] Invalid or missing image path for row {index}: {image_path}")
            feature_paths_list.append(np.nan)
            failed_count +=1
            continue

        vec = extract_resnet_features(str(image_path))

        if vec is not None:
            # Create a unique filename based on the original image filename
            base_fname = os.path.basename(str(image_path))
            fname_no_ext = os.path.splitext(base_fname)[0]
            out_filename = f"{fname_no_ext}.npy" # e.g., image_001.npy
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            
            try:
                np.save(out_path, vec)
                feature_paths_list.append(out_path)
                processed_count +=1
            except Exception as e:
                # print(f"[!] Failed to save .npy for {image_path}: {e}")
                feature_paths_list.append(np.nan)
                failed_count +=1
        else:
            feature_paths_list.append(np.nan)
            failed_count +=1

    df[feature_col_to_add] = feature_paths_list
    
    # Save a backup before overwriting
    backup_path = csv_path.replace(".csv", ".resnet50.backup.csv")
    try:
        df_existing_backup = pd.read_csv(backup_path) # if it exists, use it
        df_existing_backup[feature_col_to_add] = df[feature_col_to_add] # update only this col
        df_existing_backup.to_csv(backup_path, index=False)
    except FileNotFoundError:
        df.to_csv(backup_path, index=False) # if not, create new backup
    
    print(f"[ResNet50] Backup of CSV with new column saved to {backup_path}")

    # Overwrite the original CSV
    df.to_csv(csv_path, index=False)
    print(f"[ResNet50] Successfully processed {processed_count} images.")
    print(f"[ResNet50] Failed to process {failed_count} images.")
    print(f"[ResNet50] Updated CSV '{csv_path}' with column '{feature_col_to_add}'.")

if __name__ == "__main__":
    process_all_resnet(CSV_PATH, IMAGE_COL, FEATURE_COL_NAME)
    print("[ResNet50] Feature extraction complete.")