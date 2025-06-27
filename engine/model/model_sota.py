# /engine/model/model_sota.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, TimeDistributed, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

def build_sota_model(clinical_input_shape):
    """
    Builds the State-of-the-Art Tri-Modal model using a pre-trained EfficientNetB0 base.
    """
    INPUT_SHAPE = (224, 224, 3) # EfficientNetB0 expects this input shape
    
    # --- Input Layers ---
    topk_input = Input(shape=(5,) + INPUT_SHAPE, name="topk_input")
    global_input = Input(shape=INPUT_SHAPE, name="global_input")
    clinical_input = Input(shape=clinical_input_shape, name="clinical_input")

    # --- Load Pre-trained EfficientNetB0 Base ---
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=INPUT_SHAPE, 
        pooling='avg' # Applies Global Average Pooling at the end
    )

    # --- Fine-Tuning Strategy ---
    # Freeze most of the model, unfreeze the top blocks for fine-tuning.
    base_model.trainable = True
    # The exact number to unfreeze can be tuned. -40 is a good starting point.
    fine_tune_at = -40 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # --- Process Image Branches using the same base model ---
    patch_features = TimeDistributed(base_model)(topk_input)
    # The output is (batch, 5, features). Average across the 5 patches.
    patch_features = GlobalAveragePooling1D(name='avg_patch_features')(patch_features)

    global_features = base_model(global_input)

    # --- Clinical Branch ---
    csv_features = Dense(64, activation='relu')(clinical_input)
    csv_features = Dropout(0.4)(csv_features)
    
    # --- Fusion & Classifier Head ---
    combined_features = Concatenate()([patch_features, global_features, csv_features])
    
    x = Dense(256, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    
    output = Dense(1, activation='sigmoid', name="output")(x)

    # --- Build and Compile ---
    model = Model(inputs=[topk_input, global_input, clinical_input], outputs=output)
    
    # Fine-tuning requires a smaller learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    
    return model