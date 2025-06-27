# /engine/model/model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

# Use relative imports to get the branch-building functions
# from the sibling modules in the same 'model' package.
from .topk import build_topk_branch
from .globalv import build_global_branch
from .clinical import build_clinical_branch

def build_pinn_bcd_model(
    topk_input_shape: tuple = (5, 128, 128, 1),
    global_input_shape: tuple = (128, 128, 1),
    clinical_input_shape: tuple = (15,) # Example shape, will be determined by CSV
) -> tf.keras.Model:
    """
    Assembles the complete tri-modal network from its constituent branches.

    Args:
        topk_input_shape (tuple): Shape for the top-k patches input.
        global_input_shape (tuple): Shape for the global image input.
        clinical_input_shape (tuple): Shape for the clinical CSV data input.

    Returns:
        tf.keras.Model: The final, compiled, trainable Keras model.
    """
    print("\n--- Assembling Full Tri-Modal Model ---")
    
    # 1. Define the three input layers
    topk_input = Input(shape=topk_input_shape, name="topk_input")
    global_input = Input(shape=global_input_shape, name="global_input")
    clinical_input = Input(shape=clinical_input_shape, name="clinical_input")

    # 2. Build each branch by calling the functions from other modules
    topk_branch = build_topk_branch(topk_input_shape)
    global_branch = build_global_branch(global_input_shape)
    clinical_branch = build_clinical_branch(clinical_input_shape)

    # 3. Process the inputs through their respective branches to get feature vectors
    topk_features = topk_branch(topk_input)
    global_features = global_branch(global_input)
    clinical_features = clinical_branch(clinical_input)

    # 4. Fuse the feature vectors from all three branches
    combined_features = Concatenate(name="feature_fusion")([
        topk_features,
        global_features,
        clinical_features
    ])

    # 5. Add the final classifier head
    classifier_head = Dense(128, activation='relu', name="classifier_dense_1")(combined_features)
    classifier_head = Dropout(0.5, name="classifier_dropout")(classifier_head)
    output_layer = Dense(1, activation='sigmoid', name="output_prediction")(classifier_head)

    # 6. Create and compile the final model
    final_model = Model(
        inputs=[topk_input, global_input, clinical_input],
        outputs=output_layer,
        name="pinn_bcd_model"
    )

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\n✅ Full Tri-Modal Model Assembled and Compiled Successfully!")
    return final_model

if __name__ == '__main__':
    # Self-test the full model assembly
    print("--- Running Full Model Assembly Self-Test ---")
    
    # Use default shapes for the test
    pinn_model = build_pinn_bcd_model()
    
    # Print the model summary to visualize the full architecture
    pinn_model.summary()

    # You can also generate a plot of the model architecture
    tf.keras.utils.plot_model(pinn_model, to_file="full_model_architecture.png", show_shapes=True)
    print("\nModel architecture diagram saved to 'full_model_architecture.png'")