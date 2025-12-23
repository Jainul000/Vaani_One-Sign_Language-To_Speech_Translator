import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Configuration (must match record.py and extract_landmarks.py)
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['buds', 'spray', 'grow'])  # Must match recorded actions
no_sequences = 30  # Must match record.py
sequence_length = 30  # Must match record.py
FEATURES_PER_FRAME = 225  # 33*3 (pose) + 126 (hands)


def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                try:
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                except Exception as e:
                    print(f"Warning: Missing frame {action}/{sequence}/{frame_num}: {e}")
                    window.append(np.zeros(FEATURES_PER_FRAME))  # Pad with zeros
            sequences.append(window)
            labels.append(label_map[action])

    return np.array(sequences), to_categorical(labels).astype(int)


def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        LSTM(32),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Load and verify data
    X, y = load_data()
    print(f"\nData loaded - X shape: {X.shape}, y shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # Normalize
    mean, std = np.mean(X_train), np.std(X_train)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Create model
    model = create_model((sequence_length, FEATURES_PER_FRAME), len(actions))
    model.summary()

    # Callbacks
    callbacks = [
        TensorBoard(log_dir='Logs'),
        EarlyStopping(patience=50, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=20, verbose=1)
    ]

    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    print("\nEvaluation Metrics:")
    print("Confusion Matrix:")
    print(multilabel_confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # Save
    model.save('action.h5')
    np.savez('preprocess_params.npz', mean=mean, std=std)
    print("\nModel and preprocessing parameters saved")