# /model/topk.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling1D

def build_topk_branch(input_shape: tuple = (5, 128, 128, 1), name: str = "topk_branch") -> tf.keras.Model:
    """
    Builds the CNN branch for processing the Top-K image patches.

    This uses a TimeDistributed layer to apply the same CNN to each patch.

    Args:
        input_shape (tuple): The shape of the input stack of patches (K, H, W, C).
        name (str): The name for the overall model branch.

    Returns:
        tf.keras.Model: The Keras model for this branch.
    """
    # First, define the simple CNN that will be applied to EACH patch
    patch_cnn = Sequential([
        Input(shape=input_shape[1:]), # Input shape for a single patch: (128, 128, 1)
        Conv2D(16, (5, 5), activation='relu', padding='same', name="patch_conv_1"),
        MaxPooling2D(pool_size=(2, 2), name="patch_pool_1"),
        
        Conv2D(32, (3, 3), activation='relu', padding='same', name="patch_conv_2"),
        MaxPooling2D(pool_size=(2, 2), name="patch_pool_2"),
        
        Flatten(name="patch_flatten"),
        Dense(128, activation='relu', name="patch_dense_1") # Output a feature vector for each patch
    ], name="base_patch_cnn")
    
    # Now, define the full branch using the TimeDistributed wrapper
    # This takes the stack of 5 patches as input
    full_input = Input(shape=input_shape, name="topk_input_layer")
    
    # The TimeDistributed layer applies the 'patch_cnn' to each of the 5 time steps (patches)
    # The output will have shape (None, 5, 128) -> 5 feature vectors of size 128
    time_dist_output = TimeDistributed(patch_cnn, name="time_distributed_cnn")(full_input)
    
    # We now have 5 feature vectors. We need to aggregate them into a single
    # vector representing all patches. GlobalAveragePooling1D is a great way to do this.
    # It averages the 5 vectors, resulting in a single vector of size 128.
    aggregated_features = GlobalAveragePooling1D(name="aggregate_patch_features")(time_dist_output)
    
    # Create the final model for this branch
    model = Model(inputs=full_input, outputs=aggregated_features, name=name)
    
    print(f"✅ Top-K Branch built successfully with output shape: {model.output_shape}")
    return model

if __name__ == '__main__':
    # Self-test the branch
    print("--- Running Top-K Branch Self-Test ---")
    topk_model = build_topk_branch()
    topk_model.summary()
    # Dummy input to verify output shape
    dummy_input = tf.random.uniform(shape=(1, 5, 128, 128, 1))
    output = topk_model(dummy_input)
    print(f"Output shape on dummy data: {output.shape}")
    assert output.shape == (1, 128)
    print("✅ Self-test passed!")