# /engine/utils/callbacks.py

import tensorflow as tf
import logging
import time

class RichLoggerCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback for detailed, formatted logging at the end of each epoch.
    """
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('Trainer_V2.RichLogger')
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Record the start time of the epoch."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs detailed metrics at the end of an epoch.
        'logs' is a dictionary containing the metrics from the epoch.
        """
        epoch_time = time.time() - self.epoch_start_time
        
        # --- Safely get all the metrics from the logs dictionary ---
        # Training metrics
        train_loss = logs.get('loss', 'N/A')
        train_auc = logs.get('auc', 'N/A')
        
        # Validation metrics
        val_loss = logs.get('val_loss', 'N/A')
        val_auc = logs.get('val_auc', 'N/A')
        
        # Learning Rate (from the optimizer)
        # Accessing it this way is robust and works with schedulers.
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            # For schedulers like Adam's default decay, we get the current value
            lr = lr(self.model.optimizer.iterations)
        
        # --- Format the metrics into a clean string ---
        # The format specifiers ensure alignment and consistent decimal places
        log_string = (
            f"Epoch {epoch + 1:02d}/{self.params['epochs']} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {lr.numpy():.1e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        # Log to the file and print to the console
        self.logger.info(log_string)
        print(log_string) # Also print to terminal for real-time feedback