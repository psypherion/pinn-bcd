import pandas as pd

def preprocess_metadata(df):
    # Drop obviously non-useful columns
    drop_cols = [
        'file_path', 'image_path', 'AccessionNumber', 'PatientName', 
        'ReferringPhysicianName', 'SOPClassUID', 'SOPInstanceUID', 
        'StudyInstanceUID', 'SeriesInstanceUID', 'SecondaryCaptureDeviceManufacturer',
        'SecondaryCaptureDeviceManufacturerModelName'
    ]
    df = df.drop(columns=drop_cols)

    # Encode Laterality (R=0, L=1)
    df['Laterality'] = df['Laterality'].map({'R': 0, 'L': 1})
    
    # Fill missing with mode or -1 for categorical
    df['Laterality'] = df['Laterality'].fillna(-1)

    # Encode SeriesDescription if available
    if 'SeriesDescription' in df.columns:
        df['SeriesDescription'] = df['SeriesDescription'].astype('category').cat.codes

    # Convert dates to numeric (e.g., scan year)
    df['StudyDate'] = pd.to_datetime(df['StudyDate'], errors='coerce')
    df['scan_year'] = df['StudyDate'].dt.year.fillna(-1)

    # Normalize pixel dimensions
    for col in ['Rows', 'Columns', 'LargestImagePixelValue', 'SmallestImagePixelValue']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Final feature set
    keep_cols = ['Laterality', 'SeriesDescription', 'scan_year', 'Rows', 'Columns',
                 'LargestImagePixelValue', 'SmallestImagePixelValue']

    return df[keep_cols]
