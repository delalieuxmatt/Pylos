import tensorflow as tf
from tensorflow.keras import layers, models
import json
import numpy as np
import datetime
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ SUCCESS: Found {len(gpus)} GPU(s).")
    print(f"   Device details: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
else:
    print("\n❌ FAILURE: No GPU found. Still using CPU.")

# --- OPTIMIZED CONFIGURATION FOR 160K GAMES ---
DATASET_PATH = "src/main/training/resources/games/all_battles_1.json"
MODEL_EXPORT_PATH = "src/main/training/resources/models/"
SELECTED_PLAYERS = []

# Training hyperparameters optimized for large dataset
DISCOUNT_FACTOR = 0.99
EPOCHS = 50              # More epochs since you have more data
BATCH_SIZE = 2048        # Larger batches for stability with big dataset
N_CORES = 8
LEARNING_RATE = 0.0005   # Slightly lower LR for better convergence

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTEROP_THREADS"] = str(N_CORES)

def main():
    print("TensorFlow version:", tf.__version__)

    model = build_model()

    # Count parameters
    total_params = model.count_params()
    print(f"Total model parameters: {total_params:,}")

    # Optimizer with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0  # Prevents exploding gradients
    )

    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    # Build dataset
    inputs, targets = build_dataset(DATASET_PATH)

    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total datapoints: {len(inputs):,}")
    print(f"Score range: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"Score mean: {targets.mean():.4f}, std: {targets.std():.4f}")
    print(f"Input shape: {inputs.shape}")

    # Calculate validation set size
    val_size = int(0.15 * len(inputs))  # 15% validation
    print(f"Training samples: {len(inputs) - val_size:,}")
    print(f"Validation samples: {val_size:,}")

    # Advanced callbacks for large dataset training
    # Advanced callbacks for large dataset training
    callbacks = [
        # Early stopping with more patience for complex model
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # Save checkpoints during training - FIXED: Added .keras extension
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_EXPORT_PATH + 'checkpoint_epoch{epoch:02d}_val{val_loss:.4f}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # Learning rate schedule with warmup
        tf.keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch: LEARNING_RATE * min(1.0, (epoch + 1) / 5),
            verbose=0
        )
    ]

    print(f"\n=== STARTING TRAINING ===")
    history = model.fit(
        inputs, targets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,  # 15% for validation
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Save final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model.export(MODEL_EXPORT_PATH + timestamp)
    model.export(MODEL_EXPORT_PATH + "latest")

    # Print training summary
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Model saved to {MODEL_EXPORT_PATH}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Best validation MAE: {min(history.history['val_mae']):.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

    # Check for overfitting
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    overfit_ratio = final_val_loss / final_train_loss
    print(f"Overfit ratio (val/train): {overfit_ratio:.3f}")
    if overfit_ratio > 1.2:
        print("⚠️  Warning: Model may be overfitting. Consider:")
        print("   - Increasing dropout")
        print("   - Adding more regularization")
        print("   - Reducing model size")
    elif overfit_ratio < 1.05:
        print("✓ Model is generalizing well!")

def build_model():
    """
    Deep Residual Network for 160k games dataset
    """
    inputs = layers.Input(shape=(38,), dtype=tf.float32)

    # Initial expansion
    x = layers.Dense(512, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    # Residual Block 1 (512 units)
    residual = x
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Transition to 256
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Residual Block 2 (256 units)
    residual = x
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Final layers
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    outputs = layers.Dense(1, activation='tanh')(x)
    return models.Model(inputs=inputs, outputs=outputs)

def build_dataset(path):
    with open(path) as f:
        data = json.load(f)

    print(f"Processing {len(data)} games...")

    inputs_list = []
    targets_list = []

    for idx, game in enumerate(data):
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(data)} games...")

        if SELECTED_PLAYERS and (
                game["lightPlayer"] not in SELECTED_PLAYERS or
                game["darkPlayer"] not in SELECTED_PLAYERS
        ):
            continue

        winner = game["winner"]
        board_history = game["boardHistory"]
        n_moves = len(board_history)

        for i, board_long in enumerate(board_history):
            raw_score = winner * (DISCOUNT_FACTOR ** (n_moves - i - 1))

            board_state = []
            light_count = 0
            dark_count = 0

            for loc in range(30):
                shift = loc * 2
                val = (board_long >> shift) & 3

                if val == 1:
                    board_state.append(1.0)
                    light_count += 1
                elif val == 2:
                    board_state.append(-1.0)
                    dark_count += 1
                else:
                    board_state.append(0.0)

            board_arr = np.array(board_state, dtype=np.float32)

            light_reserves = 15 - light_count
            dark_reserves = 15 - dark_count

            light_reserves_norm = light_reserves / 15.0
            dark_reserves_norm = dark_reserves / 15.0
            reserve_diff = (light_reserves - dark_reserves) / 15.0
            material_diff = (light_count - dark_count) / 30.0

            z0_val = np.sum(board_arr[0:16]) / 16.0
            z1_val = np.sum(board_arr[16:25]) / 9.0
            z2_val = np.sum(board_arr[25:29]) / 4.0
            z3_val = board_arr[29]

            layer_features = np.array([z0_val, z1_val, z2_val, z3_val], dtype=np.float32)

            # LIGHT PERSPECTIVE
            light_input = np.concatenate([
                board_arr,
                [light_reserves_norm, dark_reserves_norm, reserve_diff, material_diff],
                layer_features
            ])
            inputs_list.append(light_input)
            targets_list.append(raw_score)

            # DARK PERSPECTIVE
            dark_board_arr = -board_arr
            z0_dark = np.sum(dark_board_arr[0:16]) / 16.0
            z1_dark = np.sum(dark_board_arr[16:25]) / 9.0
            z2_dark = np.sum(dark_board_arr[25:29]) / 4.0
            z3_dark = dark_board_arr[29]
            dark_layer_features = np.array([z0_dark, z1_dark, z2_dark, z3_dark], dtype=np.float32)

            dark_input = np.concatenate([
                dark_board_arr,
                [dark_reserves_norm, light_reserves_norm, -reserve_diff, -material_diff],
                dark_layer_features
            ])
            inputs_list.append(dark_input)
            targets_list.append(-raw_score)

    inputs_array = np.array(inputs_list, dtype=np.float32)
    targets_array = np.array(targets_list, dtype=np.float32)

    # Shuffle
    indices = np.random.permutation(len(inputs_array))
    return inputs_array[indices], targets_array[indices]

if __name__ == "__main__":
    main()