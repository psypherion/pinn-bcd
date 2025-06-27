import pandas as pd
from pathlib import Path

class ImagePathFixerSimple:
    def __init__(self, full_paths=None, cropped_paths=None, roi_paths=None):
        self.full_lookup = self._build_lookup(full_paths) if full_paths is not None else {}
        self.cropped_lookup = self._build_lookup(cropped_paths) if cropped_paths is not None else {}
        self.roi_lookup = self._build_lookup(roi_paths) if roi_paths is not None else {}

    def _build_lookup(self, paths):
        lookup = {}
        for path in paths:
            try:
                key = Path(path).parts[-2]
                lookup[key] = str(path)
            except Exception as e:
                print(f"[!] Lookup build error: {path} -> {e}")
        return lookup

    def fix_paths_in_df(self, df):
        for col, lookup in [
            ('image_file_path', self.full_lookup),
            ('cropped_image_file_path', self.cropped_lookup),
            ('ROI_mask_file_path', self.roi_lookup)
        ]:
            df[col] = df[col].apply(lambda p: self._fix_path(p, lookup))
        return df

    def _fix_path(self, path_str, lookup_dict):
        if pd.isna(path_str):
            return None
        try:
            key = Path(path_str).parts[-2]
            return lookup_dict.get(key, None)
        except Exception:
            return None
