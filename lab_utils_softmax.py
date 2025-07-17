import numpy as np
import matplotlib.pyplot as plt


def plt_softmax(z):
    """Plot a bar chart of a softmax output vector."""
    if isinstance(z, np.ndarray):
        z = np.squeeze(z)
    plt.bar(range(len(z)), z)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Softmax Output')
    plt.show()
