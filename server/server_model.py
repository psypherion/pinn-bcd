#!/usr/bin/env python3
"""
A Starlette-based async API server for model inference.
Usage: Run this script to start the server (e.g., uvicorn server.server:app --reload)
"""

import io
import asyncio
import numpy as np
import tensorflow as tf
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# Load the model (ensure you export your model as a SavedModel)
# Adjust the path as necessary.
MODEL_PATH = "model/pinn_res.keras"  # SavedModel directory.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    """Load an image, convert to RGB, resize, and normalize."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension.
    return img_tensor

async def predict(request):
    """Endpoint to accept image file upload and return model prediction."""
    form = await request.form()
    file = form.get("file")
    if file is None:
        return JSONResponse({"error": "No file provided"}, status_code=400)
    
    image_bytes = await file.read()
    # Offload image preprocessing to a thread to avoid blocking the event loop.
    img_tensor = await asyncio.to_thread(preprocess_image, image_bytes)
    
    # Run inference. Our model outputs three values; we use the first (classification).
    predictions, physics, flat = model(img_tensor, training=False)
    predictions_np = predictions.numpy()
    predicted_class = int(np.argmax(predictions_np))
    confidence = float(np.max(predictions_np))
    confidence = round(confidence, 4)*100
    # Map class index to label.
    label_map = {0: "Benign", 1: "Malignant"}
    result = {
        "predicted_class": label_map.get(predicted_class, "Unknown"),
        "confidence": f"{confidence:.2f}%",
        "predictions": predictions_np.tolist()
    }
    return JSONResponse(result)

# Define routes.
routes = [
    Route("/predict", predict, methods=["POST"])
]

# Create Starlette app.
app = Starlette(debug=True, routes=routes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
