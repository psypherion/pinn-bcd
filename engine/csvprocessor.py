import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathfixer import ImagePathFixerSimple

# === CONFIG ===
MASS_CSV_PATH = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv"
CALC_CSV_PATH = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv"
DICOM_CSV_PATH = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv"
OUTPUT_DIR = "./csv"
JPEG_DIR = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/jpeg/"
NUM_TRAIN = 1000
PCA_VARIANCE_THRESHOLD = 0.95

# === LOAD & CLEAN DICOM ===
def preprocess_dicom(df):
    df['image_path'] = df['image_path'].str.strip()
    df['Laterality'] = df['Laterality'].map({'R': 0, 'L': 1}).fillna(-1)
    df['PhotometricInterpretation'] = df['PhotometricInterpretation'].astype('category').cat.codes
    df = df.rename(columns={"PatientID": "patient_id"})
    return df[['image_path', 'Rows', 'Columns', 'Laterality', 'PhotometricInterpretation', 'SeriesDescription', 'patient_id']]

# === SHARED METADATA CLEANING ===
def preprocess_metadata(df, source_type):
    df = df.rename(columns={
        'left or right breast': 'left_or_right_breast',
        'image view': 'image_view',
        'abnormality id': 'abnormality_id',
        'abnormality type': 'abnormality_type',
        'mass shape': 'mass_shape',
        'mass margins': 'mass_margins',
        'calc type': 'calc_type',
        'calc distribution': 'calc_distribution',
        'image file path': 'image_file_path',
        'cropped image file path': 'cropped_image_file_path',
        'ROI mask file path': 'ROI_mask_file_path',
    })

    df['label'] = df['pathology'].map({'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1})
    df['label_3class'] = df['pathology'].map({'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 1, 'MALIGNANT': 2})
    df['no_callback'] = (df['pathology'] == 'BENIGN_WITHOUT_CALLBACK').astype(int)

    df['breast_side'] = df['left_or_right_breast'].map({'LEFT': 0, 'RIGHT': 1})
    df['image_view'] = df['image_view'].astype('category').cat.codes

    df['subtlety'] = df['subtlety'].fillna(df['subtlety'].mean())
    df['assessment'] = df['assessment'].fillna(df['assessment'].median())

    if source_type == "mass":
        df['mass_shape'] = df['mass_shape'].astype('category').cat.codes
        df['mass_margins'] = df['mass_margins'].astype('category').cat.codes
        df['calc_type'] = -1
        df['calc_distribution'] = -1
    elif source_type == "calc":
        df['calc_type'] = df['calc_type'].astype('category').cat.codes
        df['calc_distribution'] = df['calc_distribution'].astype('category').cat.codes
        df['mass_shape'] = -1
        df['mass_margins'] = -1

    df['type'] = source_type
    return df

def preprocess_before_pca(df, exclude_cols):
    print("[✓] Preprocessing before PCA...")

    feature_df = df.drop(columns=exclude_cols)
    cat_cols = feature_df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        print(f"[~] One-hot encoding: {cat_cols}")
        feature_df = pd.get_dummies(feature_df, columns=cat_cols)

    num_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    num_cols_valid = [col for col in num_cols if feature_df[col].notna().sum() > 0]
    dropped = set(num_cols) - set(num_cols_valid)
    if dropped:
        print(f"[~] Dropping fully-NaN columns: {list(dropped)}")
        feature_df = feature_df[num_cols_valid]

    imputer = SimpleImputer(strategy="median")
    feature_df[num_cols_valid] = imputer.fit_transform(feature_df[num_cols_valid])
    feature_df[num_cols_valid] = StandardScaler().fit_transform(feature_df[num_cols_valid])

    return pd.concat([df[exclude_cols].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

def reduce_with_pca(df, exclude_cols, threshold=0.95):
    df = preprocess_before_pca(df, exclude_cols)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"[PCA] Reducing {len(feature_cols)} numeric features...")
    pca = PCA(n_components=threshold)
    reduced = pca.fit_transform(df[feature_cols])
    reduced_df = pd.DataFrame(reduced, columns=[f"pca_{i}" for i in range(reduced.shape[1])])
    return pd.concat([df[exclude_cols].reset_index(drop=True), reduced_df], axis=1)

def assign_split_flags(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['split_train'] = 0
    df['split_valid'] = 0
    df['split_test'] = 0
    df.loc[:NUM_TRAIN-1, 'split_train'] = 1
    remainder = df.loc[NUM_TRAIN:]
    val_size = len(remainder) // 2
    df.loc[NUM_TRAIN:NUM_TRAIN + val_size - 1, 'split_valid'] = 1
    df.loc[NUM_TRAIN + val_size:, 'split_test'] = 1
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] Loading data...")
    mass_df = preprocess_metadata(pd.read_csv(MASS_CSV_PATH), source_type='mass')
    calc_df = preprocess_metadata(pd.read_csv(CALC_CSV_PATH), source_type='calc')
    dicom_df = preprocess_dicom(pd.read_csv(DICOM_CSV_PATH))

    print("[2] Getting path types from DICOM...")
    full_mammogram_paths = dicom_df[dicom_df.SeriesDescription == "full mammogram images"].image_path.str.replace("CBIS-DDSM/jpeg", JPEG_DIR, regex=False)
    cropped_image_paths = dicom_df[dicom_df.SeriesDescription == "cropped images"].image_path.str.replace("CBIS-DDSM/jpeg", JPEG_DIR, regex=False)
    roi_mask_paths = dicom_df[dicom_df.SeriesDescription == "ROI mask images"].image_path.str.replace("CBIS-DDSM/jpeg", JPEG_DIR, regex=False)

    print("[3] Concatenating datasets...")
    full_df = pd.concat([mass_df, calc_df], ignore_index=True)

    print("[4] Fixing image paths using lookup...")
    fixer = ImagePathFixerSimple(full_mammogram_paths, cropped_image_paths, roi_mask_paths)
    full_df = fixer.fix_paths_in_df(full_df)

    print("[5] Dropping rows with missing image paths...")
    full_df = full_df.dropna(subset=['image_file_path', 'cropped_image_file_path', 'ROI_mask_file_path'])
    full_df = full_df[full_df['image_file_path'].apply(os.path.exists)]
    full_df = full_df[full_df['cropped_image_file_path'].apply(os.path.exists)]
    full_df = full_df[full_df['ROI_mask_file_path'].apply(os.path.exists)].reset_index(drop=True)

    print("[6] Reducing dimensionality with PCA...")
    exclude_cols = [col for col in full_df.columns if (
        col.endswith('_file_path') or 
        'patient' in col.lower() or 
        'pathology' in col.lower() or 
        'abnormality' in col.lower() or 
        col in ['label', 'label_3class', 'no_callback', 'type']
    )]

    reduced_df = reduce_with_pca(full_df, exclude_cols=exclude_cols, threshold=PCA_VARIANCE_THRESHOLD)

    print("[7] Assigning train/val/test flags...")
    final_df = assign_split_flags(reduced_df)

    final_path = os.path.join(OUTPUT_DIR, "train.csv")
    final_df.to_csv(final_path, index=False)
    print(f"[✓] Dataset saved to: {final_path}")

if __name__ == "__main__":
    main()
