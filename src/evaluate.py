import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate():
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    model = tf.keras.models.load_model("models/model.h5")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate Confusion Matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    print("Metrics written to metrics.json and plot saved to models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
