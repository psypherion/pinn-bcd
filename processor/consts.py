import os
import pandas as pd

DATA_DIR: str = "../kaggle/data/cbis-ddsm-breast-cancer-image-dataset/"
CSV_DIR: str = DATA_DIR + "csv/"
IMAGE_DIR: str = DATA_DIR + "images/"

META_CSV: str = os.path.join(os.getcwd(), CSV_DIR, "meta.csv")
DICOM_INFO_CSV: str = os.path.join(os.getcwd(), CSV_DIR, "dicom_info.csv")
TRAIN_MASS_CSV: str = os.path.join(os.getcwd(), CSV_DIR, "mass_case_description_train_set.csv")
TRAIN_CALC_CSV: str = os.path.join(os.getcwd(), CSV_DIR, "calc_case_description_train_set.csv")
TEST_MASS_CSV: str = os.path.join(os.getcwd(), CSV_DIR, "mass_case_description_test_set.csv")
TEST_CALC_CSV: str = os.path.join(os.getcwd(), CSV_DIR, "calc_case_description_test_set.csv")


