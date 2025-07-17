import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    """Load MNIST digit images and labels."""
    (train_images, train_labels), _ = mnist.load_data()
    # Normalize images to [0, 1] and reshape
    images = train_images.astype(np.float32) / 255.0
    labels = train_labels.astype(np.int64)
    return images, labels
