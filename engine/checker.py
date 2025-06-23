# csv_sanity_checker.py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
CSV_PATH = "./csv/train.csv"
BACKUP_CSV_PATH = "./csv/train.csv.sanity_backup" # Path for backing up before cleaning

# Essential columns for the ResNet-only pipeline (after csvprocessor.py)
EXPECTED_COLUMNS_FROM_CSVPROCESSOR = [
    "image_file_path",          # For ResNet input
    "cropped_image_file_path",  # Originally for LBP, might still be in CSV
    "label_3class"              # For target labels
    # Add any other columns you expect from csvprocessor.py that are NOT features
]

# Column added by reshist.py
RESNET_FEAT_PATH_COL = "resnet50_feat_path"

# Column potentially added by the aborted lbp_generator.py (we want to remove this)
LBP_FEAT_PATH_COL = "lbp_hist_path"


def check_csv_sanity(csv_path, backup_path):
    print(f"--- Running Sanity Check for: {csv_path} ---")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found at '{csv_path}'. Cannot perform sanity check.")
        return False, None

    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV. Shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Could not read CSV file: {e}")
        return False, None

    issues_found = False
    columns_to_drop = []

    # 1. Check for essential columns from csvprocessor.py
    print("\n[1] Checking for essential columns from csvprocessor.py...")
    for col in EXPECTED_COLUMNS_FROM_CSVPROCESSOR:
        if col not in df.columns:
            print(f"  [WARNING] Essential column '{col}' is MISSING.")
            issues_found = True
        else:
            print(f"  [OK] Column '{col}' found.")
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"    [INFO] Column '{col}' has {nan_count} NaN values out of {len(df)} rows.")
                if col in ["image_file_path", "label_3class"]: # Critical NaNs
                    issues_found = True
                    print(f"      [CRITICAL] NaNs in '{col}' will cause problems for feature extraction or training.")


    # 2. Check for ResNet feature path column (from reshist.py)
    print(f"\n[2] Checking for ResNet feature path column ('{RESNET_FEAT_PATH_COL}')...")
    if RESNET_FEAT_PATH_COL not in df.columns:
        print(f"  [INFO] Column '{RESNET_FEAT_PATH_COL}' not found. This is OK if reshist.py hasn't run yet.")
    else:
        print(f"  [OK] Column '{RESNET_FEAT_PATH_COL}' found.")
        nan_count = df[RESNET_FEAT_PATH_COL].isna().sum()
        if nan_count > 0:
            print(f"    [INFO] Column '{RESNET_FEAT_PATH_COL}' has {nan_count} NaN values (rows where ResNet features might be missing or not yet processed).")
        # Check if paths actually exist (sample a few if dataframe is large)
        valid_paths_count = 0
        total_paths_to_check = 0
        for path_val in df[RESNET_FEAT_PATH_COL].dropna().sample(min(10, len(df[RESNET_FEAT_PATH_COL].dropna()))): # Check up to 10 valid paths
            if os.path.exists(str(path_val)):
                valid_paths_count += 1
            total_paths_to_check +=1
        if total_paths_to_check > 0 :
            print(f"    [INFO] Sample check: {valid_paths_count}/{total_paths_to_check} existing paths in '{RESNET_FEAT_PATH_COL}'.")


    # 3. Check for LBP feature path column (and offer to remove it)
    print(f"\n[3] Checking for LBP feature path column ('{LBP_FEAT_PATH_COL}')...")
    if LBP_FEAT_PATH_COL in df.columns:
        print(f"  [INFO] Column '{LBP_FEAT_PATH_COL}' (for LBP features) FOUND.")
        print(f"         Since LBP features are NOT being used, this column can be removed to avoid confusion.")
        nan_count = df[LBP_FEAT_PATH_COL].isna().sum()
        if nan_count < len(df):
             print(f"         It contains {len(df) - nan_count} non-NaN values (potentially from an aborted run).")
        columns_to_drop.append(LBP_FEAT_PATH_COL)
        # No need to mark as an "issue" if we intend to drop it.
    else:
        print(f"  [OK] Column '{LBP_FEAT_PATH_COL}' not found (which is good as LBP is not used).")

    # Summary of issues
    if issues_found:
        print("\n[SUMMARY] Critical issues found that might prevent subsequent scripts from running correctly.")
    else:
        print("\n[SUMMARY] Basic sanity checks passed for essential columns (excluding LBP).")

    # Offer to clean the CSV
    if columns_to_drop:
        print(f"\n--- CSV Cleaning Recommendation ---")
        print(f"The following columns are recommended for removal: {', '.join(columns_to_drop)}")
        
        user_input = input(f"Do you want to remove these columns and save a cleaned version of '{csv_path}'? (A backup will be made at '{backup_path}') [y/N]: ").strip().lower()
        if user_input == 'y':
            try:
                # Create a backup
                if os.path.exists(backup_path):
                    print(f"  [INFO] Backup file '{backup_path}' already exists. Overwriting.")
                df.to_csv(backup_path, index=False)
                print(f"  [OK] Backup of original CSV saved to: {backup_path}")

                # Drop columns
                df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
                df_cleaned.to_csv(csv_path, index=False)
                print(f"  [OK] Columns {', '.join(columns_to_drop)} removed. Cleaned CSV saved to: {csv_path}")
                print(f"  New shape of CSV: {df_cleaned.shape}")
                return True, df_cleaned # Return success and the cleaned dataframe
            except Exception as e:
                print(f"  [ERROR] Failed to clean and save CSV: {e}")
                return False, df # Return failure and original dataframe
        else:
            print("  CSV cleaning skipped by user.")
            return True, df # No cleaning done, but sanity check itself ran
            
    return not issues_found, df


if __name__ == "__main__":
    success, df_after_check = check_csv_sanity(CSV_PATH, BACKUP_CSV_PATH)
    if success:
        print(f"\nSanity check process completed. Please review the output above.")
        if df_after_check is not None:
            print(f"DataFrame (potentially cleaned) has shape: {df_after_check.shape}")
            # You can add more checks on df_after_check here if needed
    else:
        print(f"\nSanity check process identified critical issues or failed to run.")