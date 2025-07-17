import numpy as np
import matplotlib.pyplot as plt


def load_data(num_samples: int = 5000):
    """Return synthetic digit data.

    Parameters
    ----------
    num_samples : int
        Number of synthetic samples to generate.

    Returns
    -------
    X : ndarray of shape (num_samples, 400)
        Random pixel data scaled between 0 and 1.
    y : ndarray of shape (num_samples, 1)
        Random digit labels from 0 to 9.
    """
    rng = np.random.default_rng(0)
    X = rng.random((num_samples, 400), dtype=np.float32)
    y = rng.integers(0, 10, size=(num_samples, 1))
    return X, y


def display_digit(image):
    """Display a single 20x20 digit image."""
    fig, ax = plt.subplots()
    ax.imshow(image.reshape(20, 20).T, cmap="gray_r")
    ax.axis("off")
    plt.show()


def display_errors(model, X, y):
    """Return the number of misclassified samples."""
    pred = np.argmax(model.predict(X), axis=1).reshape(-1, 1)
    return int(np.sum(pred != y))


def plt_act_trio():
    """Plot linear, ReLU and sigmoid activation functions."""
    x = np.linspace(-4, 4, 200)
    plt.plot(x, x, label="linear")
    plt.plot(x, np.maximum(0, x), label="relu")
    plt.plot(x, 1 / (1 + np.exp(-x)), label="sigmoid")
    plt.legend()
    plt.show()
