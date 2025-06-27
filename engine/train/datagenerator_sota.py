# /engine/train/datagenerator_sota.py

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class SOTADataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for the SOTA model.
    - Loads 224x224 processed images.
    - Converts grayscale (1 channel) to 3-channel format for EfficientNet.
    - Includes data augmentation.
    """
    def __init__(self, df: pd.DataFrame, processed_img_dir: str, batch_size: int = 8,
                 shuffle: bool = True, augment: bool = False):
        self.logger = logging.getLogger('SOTA_DataGenerator')
        self.df = df.copy().reset_index(drop=True)
        self.processed_img_dir = processed_img_dir
        # Use a smaller batch size for large models to fit in memory
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = self.df.index.tolist()
        
        if self.augment:
            self.augmentation_pipeline = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomContrast(0.1)
            ], name="sota_augmentation")
            self.logger.info("SOTA Data augmentation is ENABLED.")
        
        self.categorical_features = ['breast_density', 'left or right breast', 'image view', 'abnormality type']
        self.preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)],
            remainder='drop'
        ).fit(self.df[self.categorical_features])
        
        self.logger.info(f"SOTA DataGenerator initialized for {len(self.indexes)} samples.")
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)
            
    def __getitem__(self, index: int) -> tuple:
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        return self._data_generation(batch_df)
            
    def _data_generation(self, batch_df: pd.DataFrame) -> tuple:
        batch_patches, batch_global_imgs, valid_indices = [], [], []

        for idx, row in batch_df.iterrows():
            try:
                img_dir = os.path.join(self.processed_img_dir, row['unique_id'])
                patches = np.load(os.path.join(img_dir, 'patches_sota.npy'))
                global_img = np.load(os.path.join(img_dir, 'global_sota.npy'))

                # --- KEY STEP: Convert 1-channel grayscale to 3-channel ---
                patches_rgb = np.repeat(patches, 3, axis=-1)
                global_img_rgb = np.repeat(global_img, 3, axis=-1)
                
                if self.augment:
                    global_img_rgb = self.augmentation_pipeline(tf.expand_dims(global_img_rgb, 0), training=True)[0].numpy()
                    augmented_patches = [self.augmentation_pipeline(tf.expand_dims(p, 0), training=True)[0].numpy() for p in patches_rgb]
                    patches_rgb = np.array(augmented_patches)

                batch_patches.append(patches_rgb)
                batch_global_imgs.append(global_img_rgb)
                valid_indices.append(idx)
            except Exception as e:
                self.logger.error(f"Error loading SOTA data for id {row.get('unique_id', 'N/A')}: {e}")
        
        valid_batch_df = batch_df.loc[valid_indices]
        batch_csv = self.preprocessor.transform(valid_batch_df[self.categorical_features])
        
        X = {'topk_input': np.array(batch_patches),
             'global_input': np.array(batch_global_imgs),
             'clinical_input': batch_csv.astype(np.float32)}
        y = valid_batch_df['label'].values
        return X, y