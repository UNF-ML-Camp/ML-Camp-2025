import numpy as np
import typing as T
import matplotlib.pyplot as plt


def plot_doodles(labels: T.List[str], n_rows: int = 5, title: str = None, doodles: np.ndarray = None) -> None:
    """
    Plot doodles
    Args:
    labels (List[str]): Liste af labels
    n_rows (int): Antal rækker og kolonner
    title (str): Titel på plot
    """
    # Vi skal hente N billeder
    N = n_rows ** 2
    N_per_label = N // len(labels)

    # Hent doodles for hvert label
    X = []
    y = []

    for label, name in enumerate(labels):
        X.append(doodles[:N_per_label].reshape(N_per_label, 28, 28))
        y.append(np.full(N_per_label, label))
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    # Tegn doodles
    fig, axs = plt.subplots(n_rows, n_rows, figsize = (n_rows, n_rows), constrained_layout = True)

    for i, (X_temp, y_temp) in enumerate(zip(X, y)):
        axs[i // n_rows, i % n_rows].imshow(X_temp, cmap='Greys')
        axs[i // n_rows, i % n_rows].set_title(f"Label: {y_temp}\n({labels[y_temp]})", fontsize = 8)
        axs[i // n_rows, i % n_rows].axis('off')
    if title is None:
        title = f"{N} doodles fra Quick, Draw! dataset"
    plt.suptitle(title)
    plt.show()