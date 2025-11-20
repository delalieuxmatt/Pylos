import tensorflow as tf
from tensorflow.keras import layers, models
import json
import numpy as np
import datetime
import os

DATASET_PATH = "resources/games/all_battles.json"
MODEL_EXPORT_PATH = "resources/models_/"
SELECTED_PLAYERS = []
DISCOUNT_FACTOR = 0.98
EPOCHS = 20
BATCH_SIZE = 512
N_CORES = 8
LEARNING_RATE = 0.001

os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTEROP_THREADS"] = str(N_CORES)

def main():
    print("TensorFlow version:", tf.__version__)

    model = build_model()

    # Use learning rate schedule for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    boards, scores = build_dataset(DATASET_PATH)

    print(f"# datapoints: {len(boards)}")
    print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"Score mean: {scores.mean():.2f}, std: {scores.std():.2f}")

    # Add early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        boards, scores,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Save the model as SavedModel, with date and time as the name
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    model.export(MODEL_EXPORT_PATH + timestamp)
    model.export(MODEL_EXPORT_PATH + "latest")

    print(f"\nModel saved to {MODEL_EXPORT_PATH}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

def build_model():
    """
    Improved model architecture with:
    - Deeper network
    - Batch normalization
    - Dropout for regularization
    """
    inputs = layers.Input(shape=(60,), dtype=tf.float32)

    # Third block
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Fourth block
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Output layer for regression (predict a single float value)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_dataset(path):
    """
    CRITICAL FIX: Generate training data from BOTH players' perspectives.

    The key insight: In a two-player zero-sum game, we need to train the model
    to evaluate positions from a consistent perspective. We do this by:
    1. For each board position, create TWO training examples
    2. One from Light's perspective (original board, positive if Light wins)
    3. One from Dark's perspective (inverted board, positive if Dark wins)
    """
    with open(path) as f:
        data = json.load(f)

    print(f"Processing {len(data)} games")

    boards = []
    scores = []

    for game_idx, game in enumerate(data):
        # Skip games that don't have both players in the selected players list
        if SELECTED_PLAYERS and (
                game["lightPlayer"] not in SELECTED_PLAYERS or
                game["darkPlayer"] not in SELECTED_PLAYERS
        ):
            continue

        winner = game["winner"]  # 1 for Light, -1 for Dark
        n_moves = len(game["boardHistory"])
        reserve_size = game['reserveSize']

        for i, board_as_long in enumerate(game["boardHistory"]):
            # Convert board to array (0 = Light, 1 = Dark)
            board_as_array = np.array(
                [(board_as_long >> j) & 1 for j in range(59, -1, -1)],
                dtype=np.float32
            )

            # Calculate discount based on proximity to end of game
            # Positions closer to the end are more certain
            discount = DISCOUNT_FACTOR ** (n_moves - i - 1)

            # Light's perspective: positive if Light wins
            light_score = winner * discount * reserve_size
            boards.append(board_as_array)
            scores.append(light_score)

            # Dark's perspective: flip the board (0->1, 1->0) and score
            # This teaches the model to evaluate from the active player's view
            dark_board = 1.0 - board_as_array
            dark_score = -winner * discount * reserve_size  # Flip the score
            boards.append(dark_board)
            scores.append(dark_score)

        if game_idx % 1000 == 0:
            print(f"Processed game {game_idx}/{len(data)}")

    boards_array = np.array(boards, dtype=np.float32)
    scores_array = np.array(scores, dtype=np.float32)

    # Shuffle the dataset to mix Light and Dark perspectives
    indices = np.random.permutation(len(boards_array))

    return boards_array[indices], scores_array[indices]


if __name__ == "__main__":
    main()