import os
import numpy as np
import pandas as pd
from . import *

class Preprocessor:
    def __init__(self):
        self.meta = pd.read_csv(META_CSV)
        self.dicom_info = pd.read_csv(DICOM_INFO_CSV)
        self.train_mass = pd.read_csv(TRAIN_MASS_CSV)
        self.train_calc = pd.read_csv(TRAIN_CALC_CSV)
        self.test_mass = pd.read_csv(TEST_MASS_CSV)
        self.test_calc = pd.read_csv(TEST_CALC_CSV)

    pass