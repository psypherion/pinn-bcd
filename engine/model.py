# engine/model.py (for end-to-end fine-tuning)
# This version is suitable and does not require further changes based on norm.py or dataset.py updates,
# as those affect data preprocessing, not the model architecture itself.

import torch
import torch.nn as nn
import torchvision.models as models

class FineTuneResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, feature_extract_only=False):
        """
        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
            feature_extract_only (bool): If True, freeze all conv layers and train only the new classifier head.
                                         If False, fine-tune more layers (or all, depending on logic below).
        """
        super().__init__()
        
        # Load a pretrained ResNet50 model using the recommended way for current torchvision
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None) # Or models.resnet50() for older torchvision

        # Set requires_grad for parameters based on feature_extract_only flag
        if feature_extract_only:
            print("Model Mode: Feature Extraction (training only classifier head).")
            for param in self.resnet.parameters():
                param.requires_grad = False
            # The new classifier head (fc layer) will have requires_grad=True by default
        else:
            print("Model Mode: Fine-Tuning (all specified layers are trainable).")
            # By default, all parameters of a newly loaded model have requires_grad=True.
            # If you wanted to freeze specific early layers during fine-tuning, you would do it here.
            # Example: Freeze first few layers
            # layers_to_freeze = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']
            # for name, child in self.resnet.named_children():
            #     if name in layers_to_freeze and name != 'fc': # don't freeze the fc if it's the original
            #         print(f"  Fine-tuning: Freezing {name}")
            #         for param in child.parameters():
            #             param.requires_grad = False
            #     elif name != 'fc': # Ensure later layers are trainable
            #         print(f"  Fine-tuning: Making {name} trainable")
            #         for param in child.parameters():
            #             param.requires_grad = True
            pass # For now, if not feature_extract_only, all layers (except potentially a new fc) start as trainable.


        # Replace the final fully connected layer (the classifier)
        num_ftrs = self.resnet.fc.in_features
        # Create a new fc layer. Parameters of this new layer will have requires_grad=True by default.
        self.resnet.fc = nn.Linear(num_ftrs, num_classes) 

        print(f"FineTuneResNet50 initialized. Output classes: {num_classes}")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Feature Extract Only (conv layers frozen): {feature_extract_only}")

    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    print("--- Testing FineTuneResNet50 ---")
    
    dummy_input = torch.randn(4, 3, 224, 224) # Batch of 4 images

    print("\n--- Test 1: Feature Extraction Mode ---")
    model_fe = FineTuneResNet50(num_classes=3, pretrained=True, feature_extract_only=True)
    # Count trainable parameters
    trainable_params_fe = sum(p.numel() for p in model_fe.parameters() if p.requires_grad)
    print(f"Feature extraction model - Trainable parameters: {trainable_params_fe}")
    # Expected: only params of resnet.fc (2049*3 + 3 = 6150)
    # assert trainable_params_fe == (model_fe.resnet.fc.in_features * 3 + 3) # Simple check for linear layer
    
    output_fe = model_fe(dummy_input)
    print(f"Feature extraction model output shape: {output_fe.shape}")


    print("\n--- Test 2: Full Fine-Tuning Mode ---")
    model_ft = FineTuneResNet50(num_classes=3, pretrained=True, feature_extract_only=False)
    trainable_params_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    total_params_ft = sum(p.numel() for p in model_ft.parameters())
    print(f"Full fine-tuning model - Trainable parameters: {trainable_params_ft} / Total: {total_params_ft}")
    assert trainable_params_ft == total_params_ft # All should be trainable (original ResNet + new FC)

    output_ft = model_ft(dummy_input)
    print(f"Full fine-tuning model output shape: {output_ft.shape}")

    print("\n--- FineTuneResNet50 Test Complete ---")