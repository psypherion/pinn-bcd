# /preprocessing/csvprocessor.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from io import StringIO
import logging

# Use relative import from sibling package 'utils'
from engine.utils.log import setup_logger

# ==============================================================================
# Single, correct definition of the ImagePathFixerSimple class
# ==============================================================================
class ImagePathFixerSimple:
    def __init__(self, all_image_paths: list):
        self.logger = logging.getLogger('CSVProcessor.PathFixer')
        self.full_lookup = self._build_lookup(all_image_paths)

    def _build_lookup(self, paths: list) -> dict:
        lookup = {}
        for path in paths:
            try:
                key = Path(path).parts[-2]
                lookup[key] = str(path)
            except Exception as e:
                self.logger.warning(f"Lookup build error on path '{path}': {e}")
        self.logger.info(f"Built path lookup dictionary with {len(lookup)} entries.")
        return lookup

    def fix_paths_in_df(self, df: pd.DataFrame, col_to_fix: str) -> pd.DataFrame:
        """Fixes the paths in a specific dataframe column."""
        self.logger.info(f"Attempting to fix paths in column '{col_to_fix}'...")
        df[col_to_fix] = df[col_to_fix].apply(lambda p: self._fix_path(p))
        return df

    def _fix_path(self, path_str: str) -> str:
        """Looks up the correct path for a single given path string."""
        if pd.isna(path_str):
            return None
        try:
            key = Path(path_str).parts[-2]
            return self.full_lookup.get(key, None)
        except Exception:
            return None

# ==============================================================================
# The rest of our processing script
# ==============================================================================

# /preprocessing/csvprocessor.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from io import StringIO
import logging

from engine.utils.log import setup_logger

# ImagePathFixerSimple class remains the same and is still essential for fixing paths
class ImagePathFixerSimple:
    def __init__(self, all_image_paths: list):
        self.logger = logging.getLogger('CSVProcessor.PathFixer')
        self.full_lookup = self._build_lookup(all_image_paths)
    def _build_lookup(self, paths: list) -> dict:
        lookup = {};
        for path in paths:
            try: key = Path(path).parts[-2]; lookup[key] = str(path)
            except Exception as e: self.logger.warning(f"Lookup build error on path '{path}': {e}")
        self.logger.info(f"Built path lookup dictionary with {len(lookup)} entries.")
        return lookup
    def fix_paths_in_df(self, df: pd.DataFrame, col_to_fix: str) -> pd.DataFrame:
        self.logger.info(f"Attempting to fix paths in column '{col_to_fix}'...")
        df[col_to_fix] = df[col_to_fix].apply(lambda p: self._fix_path(p)); return df
    def _fix_path(self, path_str: str) -> str:
        if pd.isna(path_str): return None
        try: key = Path(path_str).parts[-2]; return self.full_lookup.get(key, None)
        except Exception: return None

# Helper functions
def scan_image_directory(base_path: str, logger: logging.Logger) -> list:
    logger.info(f"Scanning for all JPEG images in: {base_path}"); pattern = os.path.join(base_path, '**', '*.jpg'); paths = glob.glob(pattern, recursive=True); logger.info(f"Found {len(paths)} total JPEG images."); return paths

def load_and_combine_training_sets(mass_path: str, calc_path: str, logger: logging.Logger) -> pd.DataFrame:
    df_mass = pd.read_csv(mass_path); df_calc = pd.read_csv(calc_path); df_calc.rename(columns={'breast density': 'breast_density'}, inplace=True); df_train_combined = pd.concat([df_mass, df_calc], ignore_index=True); logger.info(f"Mass and Calc dataframes combined. Total rows: {len(df_train_combined)}"); return df_train_combined

def engineer_features_and_drop(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Engineers final features and drops all unnecessary columns. No longer handles age."""
    # Engineer label
    df['label'] = df['pathology'].apply(lambda x: 1 if x == 'MALIGNANT' else 0)
    # Create the permanent, reliable unique ID
    df['unique_id'] = df.apply(lambda row: f"{row['patient_id']}_{row['left or right breast']}_{row['image view']}_{row.name}", axis=1)
    logger.info("Features 'label' and 'unique_id' engineered.")

    # Define all columns to drop (no longer need to drop age-related columns)
    columns_to_drop = [
        'abnormality id', 'assessment', 'subtlety', 'mass shape', 'mass margins', 'calc type',
        'calc distribution', 'cropped image file path', 'ROI mask file path', 'pathology'
    ]
    cols_dropped = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=cols_dropped)
    logger.info(f"Dropped {len(cols_dropped)} unnecessary columns.")
    return df_cleaned

def main():
    LOG_DIR = Path("logs")
    logger = setup_logger('CSVProcessor', LOG_DIR / 'csv_processing.log')
    
    DATA_DIR = Path("/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset")
    CSV_DIR = DATA_DIR / "csv"
    JPEG_DIR = DATA_DIR / "jpeg"
    OUTPUT_CSV_PATH = CSV_DIR / "test.csv"

    logger.info("--- Starting CSV Preprocessing (v6 - Age Feature Removed) ---")
    try:
        all_jpeg_paths = scan_image_directory(str(JPEG_DIR), logger)
        path_fixer = ImagePathFixerSimple(all_jpeg_paths)
        
        df_train = load_and_combine_training_sets(CSV_DIR / "mass_case_description_test_set.csv",
                                                  CSV_DIR / "calc_case_description_test_set.csv", logger)
        
        # Step 1: Fix paths
        df_train_fixed = path_fixer.fix_paths_in_df(df_train, col_to_fix='image file path')
        
        original_rows = len(df_train_fixed)
        df_train_fixed.dropna(subset=['image file path'], inplace=True)
        logger.info(f"Dropped {original_rows - len(df_train_fixed)} rows with unfixable paths.")
        
        # Step 2: Engineer features and drop columns (NO MERGE NEEDED)
        df_final = engineer_features_and_drop(df_train_fixed, logger)

        df_final.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info("--- Preprocessing Complete! ---")
        logger.info(f"Cleaned data saved to: {OUTPUT_CSV_PATH.resolve()}")
        
        buffer = StringIO()
        df_final.info(buf=buffer)
        logger.info("Final DataFrame Info:\n" + buffer.getvalue())

    except Exception as e:
        logger.exception(f"An unrecoverable error occurred: {e}")

if __name__ == "__main__":
    main()