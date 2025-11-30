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

DATASET_PATH = "resources/games/all_battles.json"
MODEL_EXPORT_PATH = "resources/models/"
SELECTED_PLAYERS = []

# Training hyperparameters optimized for large dataset
DISCOUNT_FACTOR = 0.99
EPOCHS = 40
BATCH_SIZE = 2048
N_CORES = 8
LEARNING_RATE = 0.0005

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTEROP_THREADS"] = str(N_CORES)

def main():
    print("TensorFlow version:", tf.__version__)

    model = build_larger_model()

    # Count parameters
    total_params = model.count_params()
    print(f"Total model parameters: {total_params:,}")

    # Optimizer with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
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
    val_size = int(0.15 * len(inputs))
    print(f"Training samples: {len(inputs) - val_size:,}")
    print(f"Validation samples: {val_size:,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_EXPORT_PATH + 'checkpoint_epoch{epoch:02d}_val{val_loss:.4f}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
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
        validation_split=0.15,
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

def build_larger_model():
    inputs = layers.Input(shape=(31,), dtype=tf.float32)

    board_positions = layers.Lambda(lambda x: x[:, :30])(inputs)
    reserve_diff = layers.Lambda(lambda x: x[:, 30:31])(inputs)

    board_sequence = layers.Reshape((30, 1))(board_positions)

    # Much larger Conv1D
    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(board_sequence)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=4, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(32, kernel_size=2, activation='relu', padding='same')(x)

    conv_global = layers.GlobalAveragePooling1D()(x)
    conv_max = layers.GlobalMaxPooling1D()(x)

    square_features = compute_square_features(board_positions)

    combined = layers.Concatenate()([conv_global, conv_max, square_features, reserve_diff])

    # Much larger Dense layers
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=outputs)

def build_model():
    """
    Build a Conv1D-based model for Pylos that processes the pyramid structure.

    Architecture:
    - Conv1D over the 30 positions to detect vertical patterns (stacking, layer transitions)
    - The network learns to map: 30 -> 16, 9, 4, 1 (the pyramid layers)
    - Additional square detection for horizontal patterns
    - Single global feature: reserve difference
    """
    # Input: 31 features (30 board positions + 1 reserve difference)
    inputs = layers.Input(shape=(31,), dtype=tf.float32)

    # Split input into board positions and global features
    board_positions = layers.Lambda(lambda x: x[:, :30])(inputs)
    reserve_diff = layers.Lambda(lambda x: x[:, 30:31])(inputs)

    # Reshape board positions as a sequence for Conv1D
    # Shape: (batch, 30, 1) - treating the 30 positions as a sequence
    board_sequence = layers.Reshape((30, 1))(board_positions)

    # First Conv1D: detect patterns across the full pyramid
    # Kernel size 4 can detect layer transitions and stacking patterns
    x = layers.Conv1D(64, kernel_size=4, activation='relu', padding='same')(board_sequence)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Second Conv1D: further pattern detection
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Pool to get layer-like representations
    # This naturally learns to compress the pyramid structure
    x = layers.Conv1D(16, kernel_size=2, activation='relu', padding='same')(x)

    # Global pooling to get fixed-size representation
    conv_global = layers.GlobalAveragePooling1D()(x)
    conv_max = layers.GlobalMaxPooling1D()(x)

    # Additional spatial features: detect 2x2 squares explicitly
    # This captures horizontal patterns (squares on each layer)
    square_features = compute_square_features(board_positions)

    # Combine convolutional features with square features and reserve difference
    combined = layers.Concatenate()([
        conv_global,
        conv_max,
        square_features,
        reserve_diff
    ])

    # Dense layers to process combined features
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=outputs)

def compute_square_features(board_positions):
    """
    Explicitly compute features for 2x2 squares in each layer.
    This is a key mechanic in Pylos (forming squares allows removing balls).

    Returns a feature vector indicating potential squares.
    """
    # Layer 0 (4x4): check all possible 2x2 squares
    # There are 9 possible 2x2 squares in a 4x4 grid
    layer0 = layers.Lambda(lambda x: tf.reshape(x[:, :16], [-1, 4, 4]))(board_positions)

    # Extract 2x2 patches and sum them to detect complete squares
    layer0_squares = []
    for i in range(3):
        for j in range(3):
            # Extract 2x2 square at position (i, j)
            square = layers.Lambda(
                lambda x, i=i, j=j: x[:, i:i+2, j:j+2]
            )(layer0)
            square_sum = layers.Lambda(lambda x: tf.reduce_sum(tf.abs(x), axis=[1, 2]))(square)
            layer0_squares.append(square_sum)

    layer0_square_features = layers.Concatenate()(
        [layers.Reshape((1,))(s) for s in layer0_squares]
    )

    # Layer 1 (3x3): check all possible 2x2 squares (4 squares)
    layer1 = layers.Lambda(lambda x: tf.reshape(x[:, 16:25], [-1, 3, 3]))(board_positions)

    layer1_squares = []
    for i in range(2):
        for j in range(2):
            square = layers.Lambda(
                lambda x, i=i, j=j: x[:, i:i+2, j:j+2]
            )(layer1)
            square_sum = layers.Lambda(lambda x: tf.reduce_sum(tf.abs(x), axis=[1, 2]))(square)
            layer1_squares.append(square_sum)

    layer1_square_features = layers.Concatenate()(
        [layers.Reshape((1,))(s) for s in layer1_squares]
    )

    # Layer 2 (2x2): the entire layer is one square
    layer2 = layers.Lambda(lambda x: x[:, 25:29])(board_positions)
    layer2_square_sum = layers.Lambda(lambda x: tf.reduce_sum(tf.abs(x), axis=1, keepdims=True))(layer2)

    # Combine all square features
    all_square_features = layers.Concatenate()([
        layer0_square_features,
        layer1_square_features,
        layer2_square_sum
    ])

    return all_square_features

def apply_symmetry(board_state, symmetry_type):
    """
    Apply one of 8 symmetries to the board state.
    symmetry_type: 0-7 (rot0, rot90, rot180, rot270, flip_h, flip_v, flip_d1, flip_d2)
    """
    result = np.zeros(30, dtype=np.float32)

    # Layer 0: 4x4 grid
    layer0 = board_state[0:16].reshape(4, 4)
    layer0_transformed = transform_grid(layer0, symmetry_type)
    result[0:16] = layer0_transformed.flatten()

    # Layer 1: 3x3 grid
    layer1 = board_state[16:25].reshape(3, 3)
    layer1_transformed = transform_grid(layer1, symmetry_type)
    result[16:25] = layer1_transformed.flatten()

    # Layer 2: 2x2 grid
    layer2 = board_state[25:29].reshape(2, 2)
    layer2_transformed = transform_grid(layer2, symmetry_type)
    result[25:29] = layer2_transformed.flatten()

    # Layer 3: single position (no transformation needed)
    result[29] = board_state[29]

    return result

def transform_grid(grid, symmetry_type):
    """Apply symmetry transformation to a 2D grid."""
    if symmetry_type == 0:
        return grid
    elif symmetry_type == 1:
        return np.rot90(grid, k=1)
    elif symmetry_type == 2:
        return np.rot90(grid, k=2)
    elif symmetry_type == 3:
        return np.rot90(grid, k=3)
    elif symmetry_type == 4:
        return np.fliplr(grid)
    elif symmetry_type == 5:
        return np.flipud(grid)
    elif symmetry_type == 6:
        return np.transpose(grid)
    elif symmetry_type == 7:
        return np.fliplr(np.transpose(grid))
    return grid

def build_dataset(path):
    with open(path) as f:
        data = json.load(f)

    print(f"Processing {len(data)} games...")

    inputs_list = []
    targets_list = []

    USE_SYMMETRY = True
    NUM_SYMMETRIES = 4 if USE_SYMMETRY else 1

    for idx, game in enumerate(data):
        if idx % 1000 == 0:
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
            reserve_diff = (light_reserves - dark_reserves) / 30.0

            # Apply all symmetries
            for sym in range(NUM_SYMMETRIES):
                board_sym = apply_symmetry(board_arr, sym)

                # LIGHT PERSPECTIVE
                light_input = np.concatenate([
                    board_sym,
                    [reserve_diff]
                ])
                inputs_list.append(light_input)
                targets_list.append(raw_score)

                # DARK PERSPECTIVE
                dark_board_sym = -board_sym
                dark_input = np.concatenate([
                    dark_board_sym,
                    [-reserve_diff]
                ])
                inputs_list.append(dark_input)
                targets_list.append(-raw_score)

    inputs_array = np.array(inputs_list, dtype=np.float32)
    targets_array = np.array(targets_list, dtype=np.float32)

    print(f"\n=== SYMMETRY AUGMENTATION ===")
    print(f"Symmetries applied: {NUM_SYMMETRIES}")
    print(f"Effective dataset size: {len(inputs_array):,} examples")

    # Shuffle
    indices = np.random.permutation(len(inputs_array))
    return inputs_array[indices], targets_array[indices]

if __name__ == "__main__":
    main()