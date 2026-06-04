import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models
import numpy as np

print(f"🎬 Keras Engine Backend: {keras.config.backend()}")

# 1. LOAD THE DATA
X = np.load("dataset/X_data.npy")
y = np.load("dataset/y_data.npy")

# --- THE FIX: UNBIASED SHUFFLING ---
# Create an array of index numbers and shuffle them randomly
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Reorder both features and labels using the shuffled index map
X = X[indices]
y = y[indices]

# Split into 80% Training and 20% Testing so we can verify true accuracy
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

num_classes = len(np.unique(y))

# 2. DEFINE THE STRUCTURE (Keep our efficient MVP layers)
model = models.Sequential([
    layers.Input(shape=(16, 8, 1)),
    
    layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=2),
    
    layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=2),
    
    layers.Flatten(),
    layers.Dropout(0.3), 
    layers.Dense(num_classes, activation="softmax")
])

# 3. COMPILE CONFIGURATIONS
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. TRAIN WITH VALIDATION TRACKING
print("\n🏋️‍♂️ Running Balanced Training Sequence...")
model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), # Watch how it performs on unseen data
    epochs=15,                      # Give it 15 epochs to fully digest the patterns
    batch_size=64
)

# 5. RE-EXPORT THE NEW SMART BRAIN
os.makedirs("ascii_cam_model", exist_ok=True)
model.export("ascii_cam_model/model.onnx", format="onnx")
print("\n✅ Smart model successfully trained and re-exported to ONNX!")