import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_params():
    if os.path.exists("params.yaml"):
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f).get("preprocess", {})
    return {"val_size": 0.1, "seed": 42}

def preprocess():
    params = load_params()
    val_size = params.get("val_size", 0.1)
    seed = params.get("seed", 42)

    os.makedirs("data/processed", exist_ok=True)

    X_train_raw = np.load("data/raw/X_train.npy")
    y_train_raw = np.load("data/raw/y_train.npy")
    X_test_raw = np.load("data/raw/X_test.npy")
    y_test_raw = np.load("data/raw/y_test.npy")

    # Normalize pixel values
    X_train_norm = X_train_raw.astype("float32") / 255.0
    X_test_norm = X_test_raw.astype("float32") / 255.0

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_norm, y_train_raw, test_size=val_size, random_state=seed, stratify=y_train_raw
    )

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/X_test.npy", X_test_norm)
    np.save("data/processed/y_test.npy", y_test_raw)

    print("Processed arrays saved to data/processed/")

if __name__ == "__main__":
    preprocess()
