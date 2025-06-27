# /engine/model/model_v2.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, TimeDistributed, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_pinn_bcd_model_v2(clinical_input_shape, l2_reg=0.001):
    """
    Builds the V2 Tri-Modal model with L2 Regularization and increased Dropout.

    Args:
        clinical_input_shape (tuple): The shape of the clinical data input.
        l2_reg (float): The L2 regularization factor.

    Returns:
        tf.keras.Model: The compiled V2 Keras model.
    """
    # --- Input Layers (names must match the DataGenerator) ---
    topk_input = Input(shape=(5, 128, 128, 1), name="topk_input")
    global_input = Input(shape=(128, 128, 1), name="global_input")
    clinical_input = Input(shape=clinical_input_shape, name="clinical_input")

    # Define the L2 regularizer
    regularizer = l2(l2_reg)

    # --- Branch 1: Top-K Patch CNN (with L2) ---
    patch_cnn_base = tf.keras.Sequential([
        Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizer, input_shape=(128, 128, 1)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizer),
        GlobalAveragePooling2D()
    ], name="patch_cnn_base")
    
    patch_features = TimeDistributed(patch_cnn_base)(topk_input)
    patch_features = GlobalAveragePooling1D()(patch_features)

    # --- Branch 2: Global Context CNN (with L2) ---
    global_cnn_base = tf.keras.Sequential([
        Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizer, input_shape=(128, 128, 1)),
        MaxPooling2D(),
        Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizer),
        GlobalAveragePooling2D()
    ], name="global_cnn_base")
    
    global_features = global_cnn_base(global_input)

    # --- Branch 3: Clinical Data MLP (with L2) ---
    csv_features = Dense(32, activation='relu', kernel_regularizer=regularizer)(clinical_input)

    # --- Fusion and Classifier Head (with increased Dropout and L2) ---
    combined_features = Concatenate()([patch_features, global_features, csv_features])
    
    x = Dense(128, activation='relu', kernel_regularizer=regularizer)(combined_features)
    x = Dropout(0.6)(x) # Increased Dropout from 0.5 to 0.6
    x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
    x = Dropout(0.6)(x) # Added a second Dropout layer
    
    output = Dense(1, activation='sigmoid', name="output")(x)

    # --- Build and Compile the Final Model ---
    model = Model(inputs=[topk_input, global_input, clinical_input], outputs=output)
    
    # Use the Adam optimizer and specify the AUC metric
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    
    return model