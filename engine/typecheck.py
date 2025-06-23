import pandas as pd

# === CONFIG ===
MASS_CSV_PATH = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv"
CALC_CSV_PATH = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv"
DICOM_CSV_PATH = "kaggle/data/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv"

def describe_df(df, name):
    print(f"\n=== [{name}] ===")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:5]

        print(f"➤ {col}")
        print(f"   • Type     : {dtype}")
        print(f"   • Missing  : {missing} / {len(df)}")
        print(f"   • Unique   : {unique}")
        print(f"   • Examples : {sample_vals}")
        print("")

def main():
    mass_df = pd.read_csv(MASS_CSV_PATH)
    calc_df = pd.read_csv(CALC_CSV_PATH)
    dicom_df = pd.read_csv(DICOM_CSV_PATH)

    describe_df(mass_df, "MASS TRAIN")
    describe_df(calc_df, "CALC TRAIN")
    describe_df(dicom_df, "DICOM INFO")

if __name__ == "__main__":
    main()
