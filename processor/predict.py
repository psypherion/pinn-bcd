#!/usr/bin/env python3
"""
Script to load a final model, process an input image, and output the prediction.
Usage: python predict_image.py <path_to_image>
"""

import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> tf.Tensor:
    """
    Load an image from a file, convert to RGB, resize to target_size,
    normalize pixel values, and expand dimensions for batch processing.

    Parameters:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (width, height).

    Returns:
        tf.Tensor: Preprocessed image tensor with shape (1, target_height, target_width, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension.
    return img_tensor

def predict_image(model: tf.keras.Model, image_path: str) -> None:
    """
    Process an image, run prediction using the model, and display the result.

    Parameters:
        model (tf.keras.Model): The trained Keras model.
        image_path (str): Path to the image file.
    """
    # Preprocess the image.
    img_tensor = load_and_preprocess_image(image_path)
    
    # Run the model. Our model outputs three values: classification, physics, and flat features.
    predictions, physics, flat = model(img_tensor, training=False)
    
    # Get the predicted class.
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    
    # Map the integer to a human-readable label.
    label_map = {0: "Benign", 1: "Malignant"}
    predicted_label = label_map.get(predicted_class, "Unknown")
    confidence = np.max(predictions.numpy())
    
    # Print prediction.
    print(f"Predicted class: {predicted_label} (Confidence: {confidence:.4f})")
    
    # Display the image with the prediction.
    plt.imshow(tf.squeeze(img_tensor))
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Check for image path argument.
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
    
    # Load the final model.
    # Update the path below if your model is saved elsewhere.
    model_path = "processor/model_epoch_7.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Run prediction on the input image.
    predict_image(model, image_path)
