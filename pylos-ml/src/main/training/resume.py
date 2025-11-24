import tensorflow as tf
import numpy as np
import os
import glob
import re
from train import build_dataset, DATASET_PATH, MODEL_EXPORT_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE

# --- CONFIGURATION ---
# If you want to train for *more* total epochs than originally planned, increase this.
# If you just want to finish the original 50, leave it as 50.
TOTAL_EPOCHS = 50

def get_latest_checkpoint(model_dir):
    """Finds the checkpoint file with the highest epoch number."""
    files = glob.glob(model_dir + "checkpoint_epoch*.keras")
    if not files:
        return None, 0

    # Extract epoch numbers using regex
    # Pattern matches "epoch05" or "epoch5"
    latest_file = max(files, key=lambda f: int(re.search(r'epoch(\d+)', f).group(1)))
    latest_epoch = int(re.search(r'epoch(\d+)', latest_file).group(1))

    return latest_file, latest_epoch

def main():
    print("=== RESUMING TRAINING ===")

    # 1. Find the latest checkpoint
    checkpoint_path, initial_epoch = get_latest_checkpoint(MODEL_EXPORT_PATH)

    if checkpoint_path is None:
        print("❌ No checkpoints found! Run train.py first to start from scratch.")
        return

    print(f"✅ Found checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"   Resuming from Epoch {initial_epoch + 1}") # +1 because epochs are 0-indexed

    # 2. Load the Model (Architecture + Weights + Optimizer State)
    print("   Loading model...")
    model = tf.keras.models.load_model(checkpoint_path)

    # 3. Load Dataset
    print("   Loading dataset...")
    inputs, targets = build_dataset(DATASET_PATH)

    # 4. Re-create Callbacks
    # We need to redefine callbacks to continue saving new checkpoints
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_EXPORT_PATH + 'checkpoint_epoch{epoch:02d}_val{val_loss:.4f}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # IMPORTANT: The scheduler will now receive the correct global epoch
        tf.keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch: LEARNING_RATE * min(1.0, (epoch + 1) / 5),
            verbose=0
        )
    ]

    # 5. Resume Training
    print(f"\n=== RESUMING FIT from Epoch {initial_epoch + 1} to {TOTAL_EPOCHS} ===")
    model.fit(
        inputs, targets,
        epochs=TOTAL_EPOCHS,
        initial_epoch=initial_epoch,  # <--- THIS IS THE KEY PARAMETER
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Save Final (Same as train.py)
    model.export(MODEL_EXPORT_PATH + "resumed_final")
    print("\n=== TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()