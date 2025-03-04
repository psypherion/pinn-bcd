import pandas as pd

class ImagePathFixer:
    def __init__(self, full_mammogram, cropped_images, roi_mask, correct_dir, old_base="CBIS-DDSM/jpeg"):
        """
        Initialize the fixer with image path collections and a correct base directory.
        
        :param full_mammogram: List or Series of full mammogram image paths.
        :param cropped_images: List or Series of cropped image paths.
        :param roi_mask: List or Series of ROI mask image paths.
        :param correct_dir: The correct directory to use (e.g., "../kaggle/data/cbis-ddsm-breast-cancer-image-dataset/jpeg").
        :param old_base: The old base directory string to be replaced.
        """
        self.correct_dir = correct_dir
        self.old_base = old_base
        
        # Replace the old base in all provided paths.
        self.full_mammogram = self.replace_path(full_mammogram, old_base, correct_dir)
        self.cropped_images = self.replace_path(cropped_images, old_base, correct_dir)
        self.roi_mask = self.replace_path(roi_mask, old_base, correct_dir)
        
        # Create lookup dictionaries from image file names.
        self.full_mammo_dict = self.get_image_file_name(self.full_mammogram)
        self.cropped_images_dict = self.get_image_file_name(self.cropped_images)
        self.roi_img_dict = self.get_image_file_name(self.roi_mask)

    def replace_path(self, sample, old_path, new_path):
        """
        Replace occurrences of the old path with the new path.
        Accepts either a list of strings or a Pandas Series.
        """
        if isinstance(sample, list):
            return [s.replace(old_path, new_path) for s in sample]
        elif isinstance(sample, pd.Series):
            return sample.str.replace(old_path, new_path, regex=True)
        else:
            raise ValueError("Unsupported type for sample; expected list or Pandas Series.")

    def get_image_file_name(self, data):
        """
        Build a dictionary mapping a key (derived from the image path) to the full path.
        The key is assumed to be the second-to-last element in the path.
        """
        new_dict = {}
        for path in data:
            parts = path.split('/')
            if len(parts) >= 2:
                key = parts[-2]  # Adjust key extraction if necessary.
                new_dict[key] = path
        print(f"Dictionary created with {len(new_dict.keys())} keys.")
        return new_dict

    def fix_image_path(self, df):
        """
        Update the image paths in the DataFrame using the lookup dictionaries.
        Assumptions:
          - Column index 11 uses full_mammo_dict.
          - Column index 12 uses cropped_images_dict.
          - Column index 13 uses roi_img_dict.
          
        :param df: The Pandas DataFrame (e.g., your training dataset) to update.
        :return: The updated DataFrame.
        """
        for indx in range(len(df)):
            # Process column index 11 (full mammogram image paths)
            try:
                path_11 = df.iloc[indx, 11]
                if pd.isna(path_11):
                    continue
                parts = path_11.split('/')
                if len(parts) > 2:
                    key = parts[2]  # Adjust the index if your key position changes.
                    if key in self.full_mammo_dict:
                        df.iloc[indx, 11] = self.full_mammo_dict[key]
                    else:
                        df.iloc[indx, 11] = None
                        print(f"KeyError: '{key}' not found in full_mammo_dict for row {indx}")
            except Exception as e:
                print(f"Error processing row {indx} at column 11: {e}")
            
            # Process column index 12 (cropped image paths)
            try:
                path_12 = df.iloc[indx, 12]
                if pd.isna(path_12):
                    continue
                parts = path_12.split('/')
                if len(parts) > 2:
                    key = parts[2]
                    if key in self.cropped_images_dict:
                        df.iloc[indx, 12] = self.cropped_images_dict[key]
                    else:
                        df.iloc[indx, 12] = None
                        print(f"KeyError: '{key}' not found in cropped_images_dict for row {indx}")
            except Exception as e:
                print(f"Error processing row {indx} at column 12: {e}")
            
            # Process column index 13 (ROI mask image paths)
            try:
                path_13 = df.iloc[indx, 13]
                if pd.isna(path_13):
                    continue
                parts = path_13.split('/')
                if len(parts) > 2:
                    key = parts[2]
                    if key in self.roi_img_dict:
                        df.iloc[indx, 13] = self.roi_img_dict[key]
                    else:
                        df.iloc[indx, 13] = None
                        print(f"KeyError: '{key}' not found in roi_img_dict for row {indx}")
            except Exception as e:
                print(f"Error processing row {indx} at column 13: {e}")
        return df


correct_dir = "../kaggle/data/cbis-ddsm-breast-cancer-image-dataset/jpeg"
fixer = ImagePathFixer(full_mammogram, cropped_images, roi_mask, correct_dir)
