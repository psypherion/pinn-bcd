# engine/dataset.py (for end-to-end fine-tuning)
import torch
from torch.utils.data import Dataset # Keep this top-level import
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
from tqdm import tqdm 

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dataset-specific normalization statistics calculated from train.csv
# using engine/utils/norm.py
DATASET_MEAN = [0.20559000968933105, 0.20559000968933105, 0.20559000968933105]
DATASET_STD  = [0.2500666081905365, 0.2500666081905365, 0.2500666081905365]
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class MammogramImageDataset(Dataset):
    def __init__(self, df, image_col, label_col, transform=None, image_base_dir=""):
        self.df = df.copy() 
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.image_base_dir = image_base_dir

        self.valid_indices = []
        self.actual_image_paths = [] 

        print("Verifying image paths and preparing dataset...")
        for idx in tqdm(range(len(self.df)), desc="Checking image paths"):
            img_path_val = self.df.iloc[idx][self.image_col]
            label_val = self.df.iloc[idx][self.label_col] 

            if pd.isna(img_path_val) or pd.isna(label_val):
                continue
            
            current_path_str = str(img_path_val)
            if self.image_base_dir and not os.path.isabs(current_path_str):
                full_path = os.path.join(self.image_base_dir, current_path_str)
            else:
                full_path = current_path_str

            if os.path.exists(full_path):
                self.valid_indices.append(idx) 
                self.actual_image_paths.append(full_path) 
        
        self.df_filtered = self.df.iloc[self.valid_indices].reset_index(drop=True)
        
        if len(self.actual_image_paths) != len(self.df_filtered):
            print("CRITICAL ERROR: Mismatch between actual_image_paths and df_filtered lengths.")

        if len(self.df_filtered) < len(self.df):
            print(f"Warning: Kept {len(self.df_filtered)} from {len(self.df)} original samples after checking image paths and labels.")
        
        if len(self.df_filtered) == 0:
            print("CRITICAL WARNING: Dataset is empty after filtering. Check CSV paths, image_base_dir, and label column.")


    def __len__(self):
        return len(self.df_filtered) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_full = self.actual_image_paths[idx]
        
        try:
            image = Image.open(img_name_full).convert('RGB') 
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            print(f"Error loading image {img_name_full}: {e}. This sample will be skipped by collate_fn.")
            return None 

        label_val = self.df_filtered.iloc[idx][self.label_col]
        label = torch.tensor(int(label_val), dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        if torch.isnan(image).any():
            print(f"Warning: NaN found in transformed image for path {img_name_full}. This sample will be skipped.")
            return None 

        return image, label

def get_transforms(is_train=True, target_size=(224, 224)):
    if not isinstance(DATASET_MEAN, list) or not isinstance(DATASET_STD, list) or \
       len(DATASET_MEAN) != 3 or len(DATASET_STD) != 3:
        raise ValueError("DATASET_MEAN and DATASET_STD must be lists of 3 floats. Please update them at the top of dataset.py.")

    if is_train:
        return transforms.Compose([
            transforms.Resize(int(target_size[0] * 1.12)), 
            transforms.RandomResizedCrop(target_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(10),     
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size), 
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return torch.tensor([]), torch.tensor([]) 
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    from torch.utils.data import DataLoader # <<< --- FIXED: ADDED THIS IMPORT ---

    print("--- Testing MammogramImageDataset (End-to-End) ---")
    print(f"USING NORMALIZATION: MEAN={DATASET_MEAN}, STD={DATASET_STD}")
    if DATASET_MEAN == [0.485, 0.456, 0.406]: 
        print("WARNING: Using ImageNet default normalization statistics. "
              "It is STRONGLY recommended to calculate and use your dataset-specific stats.")

    dummy_image_dir = "dummy_images_for_dataset_test"
    os.makedirs(dummy_image_dir, exist_ok=True)
    
    dummy_img1_path = os.path.join(dummy_image_dir, "dummy_image1.jpg")
    dummy_img2_path = os.path.join(dummy_image_dir, "dummy_image2.jpg")
    non_existent_path = os.path.join(dummy_image_dir, "non_existent.jpg")

    dummy_data = {
        'image_paths_col_name': [dummy_img1_path, dummy_img2_path, non_existent_path, "another_non_existent.png"],
        'labels_col_name': [0, 1, 2, 0] 
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    try:
        Image.new('RGB', (300, 300), color = 'red').save(dummy_img1_path)
        Image.new('RGB', (400, 400), color = 'green').save(dummy_img2_path)
        print(f"Created dummy image: {os.path.abspath(dummy_img1_path)}")
        print(f"Created dummy image: {os.path.abspath(dummy_img2_path)}")
    except Exception as e:
        print(f"Could not create dummy images: {e}")

    print("\nTesting Training Dataset:")
    train_transforms = get_transforms(is_train=True, target_size=(64,64)) 
    
    train_dataset = MammogramImageDataset(df=dummy_df, 
                                          image_col='image_paths_col_name', 
                                          label_col='labels_col_name', 
                                          transform=train_transforms,
                                          image_base_dir="") 
    
    print(f"Length of training dataset (after filtering): {len(train_dataset)}") 
    assert len(train_dataset) == 2, "Dataset length after filtering is not as expected."

    if len(train_dataset) > 0:
        ret_val = train_dataset[0] 
        if ret_val is not None:
            img, lbl = ret_val
            print(f"Sample 0: Image shape: {img.shape}, Label: {lbl}")
            assert img.shape == (3, 64, 64), "Image shape is not as expected."
        else:
            print("Sample 0 was None, which should not happen for valid paths after __init__ filtering.")

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_skip_none)
        try:
            num_batches_processed = 0
            for i, (images, labels) in enumerate(train_loader):
                if images.nelement() > 0: 
                    print(f"Batch {i} loaded - Images shape: {images.shape}, Labels: {labels}")
                    num_batches_processed +=1
                else:
                    print(f"Batch {i} was empty (all samples in it were None).")
                if i >= 0: break 
            assert num_batches_processed > 0, "DataLoader did not produce any valid batches."
        except Exception as e:
            print(f"Error during DataLoader iteration: {e}")

    print("\nCleaning up dummy files...")
    if os.path.exists(dummy_img1_path): os.remove(dummy_img1_path)
    if os.path.exists(dummy_img2_path): os.remove(dummy_img2_path)
    if os.path.exists(dummy_image_dir): 
        if not os.listdir(dummy_image_dir):
            os.rmdir(dummy_image_dir)
        else: 
            print(f"Warning: Dummy directory {dummy_image_dir} not empty after removing files.")
            
    print("--- MammogramImageDataset Test Complete ---")