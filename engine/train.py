# engine/train.py (for end-to-end fine-tuning)
import os
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Adjust imports based on how you run the script ---
# This assumes dataset.py and model.py are in the same 'engine' directory
# and you run `python engine/train.py` from the project root `pinn-bcd/`.
# If running `python -m engine.train`, use relative imports like `from .dataset import ...`

from dataset import MammogramImageDataset, get_transforms, collate_fn_skip_none, DATASET_MEAN, DATASET_STD
from model import FineTuneResNet50
# --- End of import adjustment ---


# --- Configuration ---
def get_config():
    parser = argparse.ArgumentParser(description="End-to-End Breast Cancer Classification Fine-Tuning")

    # --- Data Source ---
    parser.add_argument('--csv_path', type=str, default='./csv/train.csv',
                        help="Path to the master CSV file (from csvprocessor.py)")
    parser.add_argument('--image_col', type=str, default='image_file_path',
                        help="Column name in CSV for image paths")
    parser.add_argument('--label_col', type=str, default='label_3class',
                        help="Column name in CSV for labels")
    parser.add_argument('--image_base_dir', type=str, default="",
                        help="Base directory for image paths if they are relative in CSV (e.g., './kaggle/data/.../jpeg/'). Empty if paths are absolute or correct from project root.")

    # --- Experiment Setup ---
    parser.add_argument('--experiment_name', type=str, 
                        default=f"resnet50_finetune_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Name for this experiment run")
    parser.add_argument('--base_output_dir', type=str, default='./training_output_finetune',
                        help="Base directory for all training outputs")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    # --- Model Specific ---
    parser.add_argument('--num_classes', type=int, default=3, help="Number of output classes")
    parser.add_argument('--feature_extract_only', action='store_true',
                        help="Train only the classifier head, freeze ResNet backbone layers.")
    parser.add_argument('--no-pretrained_weights', action='store_false', dest='use_pretrained_weights',
                        help="Disable loading of ImageNet pretrained weights for ResNet (Not recommended for fine-tuning).")
    parser.set_defaults(use_pretrained_weights=True)


    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Batch size (adjust based on GPU memory)")
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help="Base learning rate (for backbone if fine-tuning all, or for classifier if feature_extract_only)")
    parser.add_argument('--lr_classifier_head', type=float, default=1e-3, 
                        help="Specific learning rate for the new classifier head (if different from backbone LR)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help="Weight decay for AdamW optimizer")
    
    # --- Schedulers and Early Stopping ---
    parser.add_argument('--patience_early_stopping', type=int, default=10, 
                        help="Patience for early stopping (epochs)")
    parser.add_argument('--patience_lr_scheduler', type=int, default=3, 
                        help="Patience for ReduceLROnPlateau scheduler (epochs)")
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, 
                        help="Factor by which LR is reduced by ReduceLROnPlateau")
    
    # --- Data Handling and Augmentation ---
    parser.add_argument('--apply_class_weights', action='store_true', default=True, # Defaulting to True
                        help="Apply class weighting (default: True).")
    parser.add_argument('--no-apply-class-weights', action='store_false', dest='apply_class_weights') # To disable
    parser.set_defaults(apply_class_weights=True) # Ensures it's true if no flag is given

    parser.add_argument('--num_workers', type=int, default=2, 
                        help="Number of workers for DataLoader (adjust based on CPU cores)")
    parser.add_argument('--validation_split_ratio', type=float, default=0.2, 
                        help="Fraction of data to use for validation")
    parser.add_argument('--img_size_h', type=int, default=224, help="Target image height for resizing")
    parser.add_argument('--img_size_w', type=int, default=224, help="Target image width for resizing")

    args = parser.parse_args()

    # --- Post-argument processing ---
    args.target_image_size = (args.img_size_h, args.img_size_w)
    args.output_dir = os.path.join(args.base_output_dir, args.experiment_name)
    args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    args.plot_dir = os.path.join(args.output_dir, 'plots')
    args.results_dir = os.path.join(args.output_dir, 'results')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    return args

# --- Helper Functions (set_seed, save_checkpoint, plot_training_curves, plot_confusion_matrix_custom) ---
# These are identical to the previous 'final' train.py. For brevity, not repeating them here.
# Ensure they are present in your actual train.py file.
def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar", best_filename="best_model.pt"):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state['model_state_dict'], best_filepath)

def plot_training_curves(history, plot_dir_path): # Make sure plot_dir_path is used
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color = 'tab:red'; ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], label='Train Loss', color=color, linestyle='--')
    ax1.plot(history['val_loss'], label='Validation Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color); ax1.legend(loc='upper left'); ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax2 = ax1.twinx(); color = 'tab:blue'; ax2.set_ylabel('Accuracy / F1', color=color)
    ax2.plot(history['train_acc'], label='Train Accuracy', color=color, linestyle='--')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color=color)
    if 'val_f1_macro' in history: ax2.plot(history['val_f1_macro'], label='Validation F1 Macro', color='tab:green', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color); ax2.legend(loc='upper right')
    if 'lr' in history and history['lr']:
        ax3 = ax1.twinx(); ax3.spines["right"].set_position(("outward", 60)); color = 'tab:purple'
        ax3.set_ylabel('Learning Rate', color=color)
        ax3.plot(history['lr'], label='Learning Rate', color=color, linestyle=':')
        ax3.tick_params(axis='y', labelcolor=color); ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=1)
    fig.tight_layout(); plt.title('Training and Validation Metrics vs. Epochs')
    plt.savefig(os.path.join(plot_dir_path, 'training_curves_detailed.png'), bbox_inches='tight'); plt.close()


def plot_confusion_matrix_custom(cm, class_names, plot_dir_path, filename="confusion_matrix.png", title='Confusion Matrix'):
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.tight_layout(); plt.savefig(os.path.join(plot_dir_path, filename)); plt.close()


# --- Main Training Function ---
def main():
    config = get_config()
    set_seed(config.seed)
    
    # Save the exact configuration used for this run
    with open(os.path.join(config.results_dir, 'config_used.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)
    print("--- Configuration Used ---")
    for key, value in vars(config).items(): print(f"{key}: {value}")
    print("-------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Main DataFrame ---
    print("\n--- Loading Main DataFrame ---")
    try:
        master_df = pd.read_csv(config.csv_path)
        # Ensure essential columns are present
        if not all(col in master_df.columns for col in [config.image_col, config.label_col]):
            raise ValueError(f"CSV must contain '{config.image_col}' and '{config.label_col}' columns.")
        master_df = master_df.dropna(subset=[config.image_col, config.label_col])
    except FileNotFoundError:
        print(f"ERROR: Master CSV not found at {config.csv_path}")
        return
    except ValueError as ve:
        print(f"ERROR: {ve}")
        return
        
    print(f"Loaded {len(master_df)} samples from CSV after dropping rows with missing essential data.")
    if len(master_df) == 0:
        print("ERROR: No valid data to train on after loading CSV. Exiting.")
        return

    # --- Split DataFrame for Train/Validation ---
    train_df, val_df = train_test_split(
        master_df,
        test_size=config.validation_split_ratio,
        stratify=master_df[config.label_col], # Stratify by labels
        random_state=config.seed
    )
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    if len(train_df) == 0 or len(val_df) == 0:
        print("ERROR: Training or validation set is empty after split. Check data and split ratio.")
        return

    # --- Create Datasets and DataLoaders ---
    print("\n--- Creating Datasets and DataLoaders ---")
    # DATASET_MEAN and DATASET_STD are imported from dataset.py module
    print(f"Using image normalization: MEAN={DATASET_MEAN}, STD={DATASET_STD} (from dataset.py)")
    
    train_transforms = get_transforms(is_train=True, target_size=config.target_image_size)
    val_transforms = get_transforms(is_train=False, target_size=config.target_image_size)

    train_dataset = MammogramImageDataset(df=train_df, image_col=config.image_col, label_col=config.label_col,
                                          transform=train_transforms, image_base_dir=config.image_base_dir)
    val_dataset = MammogramImageDataset(df=val_df, image_col=config.image_col, label_col=config.label_col,
                                        transform=val_transforms, image_base_dir=config.image_base_dir)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: Training or validation dataset is empty after MammogramImageDataset initialization. Check image paths.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
                            collate_fn=collate_fn_skip_none)

    # --- Initialize Model ---
    print("\n--- Initializing Model ---")
    model = FineTuneResNet50(
        num_classes=config.num_classes,
        pretrained=config.use_pretrained_weights, # Use flag from config
        feature_extract_only=config.feature_extract_only
    ).to(device)

    # --- Define Loss, Optimizer, Scheduler ---
    class_weights = None
    if config.apply_class_weights:
        class_counts = np.bincount(train_df[config.label_col].astype(int), minlength=config.num_classes)
        total_train_samples_for_weighting = len(train_df) # Use actual length of train_df
        
        class_weights_val = []
        for count_val in class_counts: # Renamed 'count' to 'count_val' to avoid conflict
            if count_val > 0: 
                weight = total_train_samples_for_weighting / (config.num_classes * count_val)
                class_weights_val.append(weight)
            else: 
                class_weights_val.append(1.0) # Default weight if class is missing in train_df
        class_weights = torch.tensor(class_weights_val, dtype=torch.float).to(device)
        print(f"Applying 'balanced' class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Setup optimizer with potentially differential LRs
    if config.feature_extract_only:
        # If only extracting features, optimize only the classifier's parameters
        params_to_optimize = model.resnet.fc.parameters()
        optimizer = optim.AdamW(params_to_optimize, lr=config.lr_classifier_head, weight_decay=config.weight_decay)
        print(f"Optimizing only classifier head with LR: {config.lr_classifier_head}")
    else:
        # Fine-tuning more layers: separate backbone and classifier parameters
        if hasattr(model.resnet, 'fc') and isinstance(model.resnet.fc, nn.Linear):
            # Filter parameters that require gradients
            resnet_backbone_params = [p for name, p in model.resnet.named_parameters() if 'fc' not in name and p.requires_grad]
            classifier_head_params = [p for p in model.resnet.fc.parameters() if p.requires_grad] # Should all require grad

            optimizer_params_list = [
                {'params': resnet_backbone_params, 'lr': config.learning_rate}, # Base LR for backbone
                {'params': classifier_head_params, 'lr': config.lr_classifier_head} # Specific LR for new head
            ]
            optimizer = optim.AdamW(optimizer_params_list, lr=config.learning_rate, weight_decay=config.weight_decay) # Base LR is default for AdamW
            print(f"Using differential LRs: Backbone LR={config.learning_rate}, Classifier Head LR={config.lr_classifier_head}")
        else:
            # Fallback: if model structure is unexpected, optimize all trainable params with base LR
            params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.AdamW(params_to_optimize, lr=config.learning_rate, weight_decay=config.weight_decay)
            print("Optimizing all trainable model parameters with base learning rate.")

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=config.lr_scheduler_factor, 
                                  patience=config.patience_lr_scheduler)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 
               'val_f1_macro': [], 'val_auc_macro_ovr': [], 'lr': []}
    best_val_metric = 0.0 
    epochs_no_improve = 0
    class_names = ['BenignNoCB', 'Benign', 'Malignant'] 

    for epoch in range(config.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        lrs_logged = [pg['lr'] for pg in optimizer.param_groups]
        history['lr'].append(lrs_logged[0]) 
        lr_log_str = ", ".join([f"{lr:.1e}" for lr in lrs_logged])

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train LR:{lr_log_str}]")
        batch_count = 0
        for images, labels_batch in train_pbar:
            if not images.numel(): # Skip if batch is empty from collate_fn
                # print(f"Skipping empty batch in epoch {epoch+1}")
                continue 
            batch_count +=1
            images, labels_batch = images.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels_batch.size(0)
            correct_train += (predicted == labels_batch).sum().item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(predicted == labels_batch).sum().item()/labels_batch.size(0) if labels_batch.size(0) > 0 else 0:.4f}"})

        if batch_count == 0 and len(train_loader) > 0: # All batches in epoch were empty
            print(f"WARNING: Epoch {epoch+1} had no processable training batches. Check data loading and collate_fn.")
            # Decide how to handle: continue, break, or error
            # For now, let's record zero metrics for this epoch and continue
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
        elif total_train > 0:
            epoch_train_loss = running_loss / total_train
            epoch_train_acc = correct_train / total_train
        else: # No training samples processed at all (e.g. len(train_loader) was 0)
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
            
        history['train_loss'].append(epoch_train_loss); history['train_acc'].append(epoch_train_acc)

        # Validation
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        all_val_preds_epoch, all_val_labels_epoch, all_val_probs_epoch = [], [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val  ]")
        val_batch_count = 0
        with torch.no_grad():
            for images, labels_batch in val_pbar:
                if not images.numel(): # Skip if batch is empty
                    # print(f"Skipping empty validation batch in epoch {epoch+1}")
                    continue
                val_batch_count += 1
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.softmax(outputs, dim=1)
                total_val += labels_batch.size(0)
                correct_val += (predicted == labels_batch).sum().item()
                all_val_preds_epoch.extend(predicted.cpu().numpy())
                all_val_labels_epoch.extend(labels_batch.cpu().numpy())
                all_val_probs_epoch.extend(probabilities.cpu().numpy())
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(predicted == labels_batch).sum().item()/labels_batch.size(0) if labels_batch.size(0) > 0 else 0:.4f}"})

        if val_batch_count == 0 and len(val_loader) > 0 :
             print(f"WARNING: Epoch {epoch+1} had no processable validation batches.")
             epoch_val_loss, epoch_val_acc, epoch_val_f1_macro, epoch_val_auc_macro_ovr = 0.0, 0.0, 0.0, 0.0
        elif total_val > 0:
            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = correct_val / total_val
            epoch_val_f1_macro = f1_score(all_val_labels_epoch, all_val_preds_epoch, average='macro', zero_division=0)
            try: epoch_val_auc_macro_ovr = roc_auc_score(all_val_labels_epoch, np.array(all_val_probs_epoch), average='macro', multi_class='ovr')
            except ValueError: epoch_val_auc_macro_ovr = 0.0 # Happens if only one class present in predictions for a batch
        else: # No validation samples processed
            epoch_val_loss, epoch_val_acc, epoch_val_f1_macro, epoch_val_auc_macro_ovr = 0.0, 0.0, 0.0, 0.0
            
        history['val_loss'].append(epoch_val_loss); history['val_acc'].append(epoch_val_acc)
        history['val_f1_macro'].append(epoch_val_f1_macro); history['val_auc_macro_ovr'].append(epoch_val_auc_macro_ovr)
        
        # Step scheduler based on validation F1
        if total_val > 0: # Only step scheduler if validation occurred
            scheduler.step(epoch_val_f1_macro)

        print(f"Epoch {epoch+1:03d}/{config.epochs:03d} | Tr_Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val_Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {epoch_val_f1_macro:.4f} AUC: {epoch_val_auc_macro_ovr:.4f} | "
              f"LR: {lr_log_str}")

        current_metric_for_best = epoch_val_f1_macro
        is_best = current_metric_for_best > best_val_metric
        
        if is_best and total_val > 0: # Save best model only if validation occurred and metric improved
            best_val_metric = current_metric_for_best
            epochs_no_improve = 0
            print(f"    => New best val F1-macro: {best_val_metric:.4f} at epoch {epoch+1}. Saving model...")
            save_checkpoint({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 
                'best_val_metric': best_val_metric, 'config': vars(config)
            }, is_best, config.checkpoint_dir)
        elif total_val > 0: # Only increment no_improve if validation occurred
            epochs_no_improve += 1
            
        if total_val > 0 and epochs_no_improve >= config.patience_early_stopping:
            print(f"\nEarly stopping at epoch {epoch+1} due to no improvement in validation F1-macro for {config.patience_early_stopping} epochs.")
            break
        elif total_val == 0 and epoch > 5: # Stop if no val data for several epochs (arbitrary 5)
            print(f"\nStopping early at epoch {epoch+1} due to no processable validation data.")
            break
            
    # --- Post Training ---
    print("\n--- Training Finished ---")
    print(f"Best Validation F1-macro: {best_val_metric:.4f}")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config.results_dir, "training_history.csv"), index_label="epoch")
    plot_training_curves(history, config.plot_dir) # Use config.plot_dir
    print(f"Training curves saved to {config.plot_dir}")

    print("\n--- Final Evaluation on Validation Set (Using Best Model) ---")
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        # Re-initialize model for evaluation to ensure correct architecture from config
        eval_model = FineTuneResNet50(num_classes=config.num_classes, pretrained=False, 
                                      feature_extract_only=config.feature_extract_only).to(device)
        try:
            eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
            eval_model.eval()
            final_preds, final_labels = [], []
            with torch.no_grad():
                for images, labels_batch in tqdm(val_loader, desc="Best Model Validation"):
                    if not images.numel(): continue
                    images, labels_batch = images.to(device), labels_batch.to(device)
                    outputs = eval_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    final_preds.extend(predicted.cpu().numpy())
                    final_labels.extend(labels_batch.cpu().numpy())
            
            if final_labels: # Check if any validation samples were processed
                report_str = classification_report(final_labels, final_preds, target_names=class_names, zero_division=0)
                # Guard against empty predictions/labels for DataFrame creation
                try:
                    report_df = pd.DataFrame(classification_report(final_labels, final_preds, target_names=class_names, output_dict=True, zero_division=0)).transpose()
                    report_df.to_csv(os.path.join(config.results_dir, "best_model_classification_report.csv"))
                except Exception as e_rep:
                    print(f"Could not generate/save classification report DataFrame: {e_rep}")

                with open(os.path.join(config.results_dir, "best_model_classification_report.txt"), 'w') as f:
                    f.write(report_str)
                print("\nClassification Report (Best Model on Validation Set):\n", report_str)
                
                cm = confusion_matrix(final_labels, final_preds, labels=list(range(config.num_classes)))
                plot_confusion_matrix_custom(cm, class_names, config.plot_dir, "best_model_confusion_matrix.png", "CM (Best Model on Val Set)") # Use config.plot_dir
                print(f"Best model CM saved to {config.plot_dir}")
            else:
                print("No samples were processed in the final validation (all batches might have been empty). Skipping report.")
        except Exception as e_load:
            print(f"Error loading best model or during final evaluation: {e_load}")
    else: 
        print(f"Best model checkpoint ('{best_model_path}') not found. Skipping final evaluation.")
    print(f"\nAll training artifacts saved in: {config.output_dir}")

if __name__ == "__main__":
    main()