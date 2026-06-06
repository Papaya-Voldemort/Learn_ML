import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models
import numpy as np

def main():
    # 1. LOAD THE DATA
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X = np.load(os.path.join(script_dir, "dataset", "X_data.npy"))
    y = np.load(os.path.join(script_dir, "dataset", "y_data.npy"))
    
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
    
    model = models.Sequential([
        layers.Input(shape=(16, 8, 1)),
        
        layers.Conv2D(16, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 1)),
        
        layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\nRunning Training Sequence...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=15, 
        batch_size=64
    )
    
    model_dir = os.path.join(os.path.dirname(script_dir), "ascii_cam_model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.keras"))
    model.export(os.path.join(model_dir, "model.onnx"), format="onnx")

if __name__ == "__main__":
    main()