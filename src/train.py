import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

def load_params():
    if os.path.exists("params.yaml"):
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f).get("train", {})
    return {
        "dense_units": 128,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64
    }

def train():
    params = load_params()
    os.makedirs("models", exist_ok=True)

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(params["dense_units"], activation="relu"),
        layers.Dropout(params["dropout_rate"]),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )

    model.save("models/model.h5")
    pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
    print("Trained model saved to models/model.h5 and history saved to models/history.csv")

if __name__ == "__main__":
    train()
