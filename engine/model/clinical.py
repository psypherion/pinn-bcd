# /model/clinical.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_clinical_branch(input_shape: tuple, name: str = "clinical_branch") -> tf.keras.Model:
    """
    Builds the MLP branch for processing tabular clinical data.

    Args:
        input_shape (tuple): The shape of the input vector (e.g., (15,)).
        name (str): The name for the sequential model.

    Returns:
        tf.keras.Model: The Keras model for this branch.
    """
    model = Sequential([
        Input(shape=input_shape, name="csv_input_layer"),
        Dense(32, activation='relu', name="clinical_dense_1"),
        Dense(64, activation='relu', name="clinical_dense_2")
    ], name=name)
    
    print(f"✅ Clinical Branch built successfully with output shape: {model.output_shape}")
    return model

if __name__ == '__main__':
    # Self-test the branch
    print("--- Running Clinical Branch Self-Test ---")
    # Example: मान लीजिए हमारे पास वन-हॉट एन्कोडिंग के बाद 15 CSV सुविधाएँ हैं
    test_input_shape = (15,)
    clinical_model = build_clinical_branch(test_input_shape)
    clinical_model.summary()
    # Dummy input to verify output shape
    dummy_input = tf.random.uniform(shape=(1, test_input_shape[0]))
    output = clinical_model(dummy_input)
    print(f"Output shape on dummy data: {output.shape}")
    assert output.shape == (1, 64)
    print("✅ Self-test passed!")