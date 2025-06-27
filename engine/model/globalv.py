# /model/global_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def build_global_branch(input_shape: tuple = (128, 128, 1), name: str = "global_branch") -> tf.keras.Model:
    """
    Builds the CNN branch for processing the downscaled global image.

    Args:
        input_shape (tuple): The shape of the input image (H, W, C).
        name (str): The name for the sequential model.

    Returns:
        tf.keras.Model: The Keras model for this branch.
    """
    model = Sequential([
        Input(shape=input_shape, name="global_input_layer"),
        Conv2D(16, (5, 5), activation='relu', padding='same', name="global_conv_1"),
        MaxPooling2D(pool_size=(2, 2), name="global_pool_1"),
        
        Conv2D(32, (3, 3), activation='relu', padding='same', name="global_conv_2"),
        MaxPooling2D(pool_size=(2, 2), name="global_pool_2"),

        Conv2D(64, (3, 3), activation='relu', padding='same', name="global_conv_3"),
        MaxPooling2D(pool_size=(2, 2), name="global_pool_3"),
        
        Flatten(name="global_flatten"),
        Dense(64, activation='relu', name="global_dense_1")
    ], name=name)
    
    print(f"✅ Global Branch built successfully with output shape: {model.output_shape}")
    return model

if __name__ == '__main__':
    # Self-test the branch
    print("--- Running Global Branch Self-Test ---")
    global_model = build_global_branch()
    global_model.summary()
    # Dummy input to verify output shape
    dummy_input = tf.random.uniform(shape=(1, 128, 128, 1))
    output = global_model(dummy_input)
    print(f"Output shape on dummy data: {output.shape}")
    assert output.shape == (1, 64)
    print("✅ Self-test passed!")