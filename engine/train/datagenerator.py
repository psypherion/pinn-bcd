# /engine/train/datagenerator.py

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras, loading processed images and handling CSV features.
    Includes an option for data augmentation on the training set.
    """
    def __init__(self, df: pd.DataFrame, processed_img_dir: str, batch_size: int = 32,
                 shuffle: bool = True, augment: bool = False):
        """
        Initialization.
        
        Args:
            df (pd.DataFrame): The main dataframe with paths and labels.
            processed_img_dir (str): Path to the directory of processed .npy files.
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the data at the end of each epoch.
            augment (bool): Whether to apply data augmentation. Should be True for the
                            training generator and False for the validation generator.
        """
        self.logger = logging.getLogger('DataGenerator')
        self.df = df.copy().reset_index(drop=True)
        self.processed_img_dir = processed_img_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = self.df.index.tolist()
        
        # --- Define the data augmentation pipeline ---
        if self.augment:
            # This pipeline will be applied ONLY if augment=True
            self.augmentation_pipeline = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05), # Reduced rotation to 5% (about 18 degrees)
                tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
                tf.keras.layers.RandomContrast(0.1)
            ], name="augmentation_pipeline")
            self.logger.info("Data augmentation is ENABLED for this generator instance.")
        else:
            self.logger.info("Data augmentation is DISABLED for this generator instance.")
        
        # Define features for the ColumnTransformer
        self.categorical_features = ['breast_density', 'left or right breast', 'image view', 'abnormality type']
        
        # The preprocessor for one-hot encoding the CSV data
        self.preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)],
            remainder='drop'
        )
        
        self.preprocessor.fit(self.df[self.categorical_features])
        
        self.logger.info(f"DataGenerator initialized for {len(self.indexes)} samples.")
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index: int) -> tuple:
        """Generate one batch of data."""
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_index:end_index]
        batch_df = self.df.iloc[batch_indexes]
        X, y = self._data_generation(batch_df)
        return X, y
            
    def _data_generation(self, batch_df: pd.DataFrame) -> tuple:
        """Generates the data containing batch_size samples."""
        batch_patches, batch_global_imgs, valid_indices = [], [], []

        for idx, row in batch_df.iterrows():
            try:
                img_dir = os.path.join(self.processed_img_dir, row['unique_id'])
                patches_path = os.path.join(img_dir, 'patches.npy')
                global_path = os.path.join(img_dir, 'global.npy')
                
                if not (os.path.exists(patches_path) and os.path.exists(global_path)):
                    self.logger.warning(f"Files not found for id: {row['unique_id']}. Skipping.")
                    continue
                
                patches = np.load(patches_path)
                global_img = np.load(global_path)

                # --- Apply augmentation if this generator has it enabled ---
                if self.augment:
                    # The augmentation layers expect a batch dimension, so we add and remove it.
                    # We do this for the global image first.
                    g_aug = self.augmentation_pipeline(tf.expand_dims(global_img, 0), training=True)
                    global_img = g_aug[0].numpy() # Convert back to numpy array

                    # And for each of the 5 patches.
                    augmented_patches = []
                    for i in range(patches.shape[0]):
                        p_aug = self.augmentation_pipeline(tf.expand_dims(patches[i], 0), training=True)
                        augmented_patches.append(p_aug[0].numpy())
                    patches = np.array(augmented_patches)

                batch_patches.append(patches)
                batch_global_imgs.append(global_img)
                valid_indices.append(idx)

            except Exception as e:
                self.logger.error(f"Error loading/augmenting data for id {row.get('unique_id', 'N/A')}: {e}")
        
        if not valid_indices:
            return {}, np.array([])

        valid_batch_df = batch_df.loc[valid_indices]
        
        csv_data_to_transform = valid_batch_df[self.categorical_features]
        batch_csv_processed = self.preprocessor.transform(csv_data_to_transform)
        
        batch_labels = valid_batch_df['label'].values
        X = {'topk_input': np.array(batch_patches),
             'global_input': np.array(batch_global_imgs),
             'clinical_input': batch_csv_processed}
        return X, batch_labels