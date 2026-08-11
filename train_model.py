import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse

parser = argparse.ArgumentParser(description='Train a simple MLP on hand landmark data.')
parser.add_argument('--data', type=str, default='./data', help='Data directory (one subfolder per label)')
parser.add_argument('--output', type=str, default='./model/sign_model.h5', help='Output model path')
parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
parser.add_argument('--batch', type=int, default=32, help='Batch size')
args = parser.parse_args()

DATA_DIR = Path(args.data)
labels = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
if not labels:
    raise RuntimeError('No label folders found in data directory.')

print('Found labels:', labels)
label_to_idx = {l:i for i,l in enumerate(labels)}

X = []
y = []
for label in labels:
    folder = DATA_DIR / label
    for f in folder.glob('*.npy'):
        arr = np.load(f)
        X.append(arr)
        y.append(label_to_idx[label])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Check shapes
print('X shape:', X.shape)
print('y shape:', y.shape)

# If landmarks saved as (63,), ensure consistent shape
n_features = X.shape[1]

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Build model (MLP)
from tensorflow.keras import regularizers
model = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint(args.output, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
]

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=args.epochs, batch_size=args.batch, callbacks=callbacks)

print('Training finished. Best model saved to', args.output)
