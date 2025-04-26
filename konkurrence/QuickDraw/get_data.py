import os
import requests
import typing as T

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# set up data
### lav dataloader for dem
def get_doodles(name: str, verbose: bool = False) -> torch.tensor:
    """
    Downloader billeder for et bestemt label fra Quick, Draw! dataset
    Args:
    name (str): Navnet på label
    verbose (bool): Hvis True, printes status undervejs
    Returns:
    torch.tensor: Billeder for label
    """
    if os.getcwd().split('/')[-1] == 'UNF_MLCamp2024':
        path = f'2.NN/QuickDraw/data/{name}.npy'
    elif os.getcwd().split('/')[-1] == '2.NN':
        path = f'QuickDraw/data/{name}.npy'
    else:
        path = f'data/{name}.npy'
    
    # Check om data folderen allerede eksisterer
    if not os.path.exists(''.join(path.split('/')[:-1])):
        os.makedirs(''.join(path.split('/')[:-1]))

    # Check om filen allerede eksisterer
    if not os.path.exists(path):
        # Første gang vi downloader et label, skal vi hente filen fra internettet
        if verbose:
            print(f'Downloading {name}...')
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy'
        r = requests.get(url, stream = True)

        # Gem filen lokalt
        with open(path, 'wb') as f:
            f.write(r.content)

    doodles = np.load(path)
    N_doodles = doodles.shape[0]
    
    # Indlæs filen
    return torch.Tensor(doodles.reshape(N_doodles, 28, 28)).float()

def get_dataset(
    names: T.List[str],
    n_samples: int = 30000,
    seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0,
    batch_size: int = 64,
    verbose: bool = False,) -> torch.utils.data.Dataset:
    """
    Indlæser et dataset bestående af doodles fra Quick, Draw! dataset
    og splitter det i trænings-, validerings- og test-sæt
    Args:
    names (List[str]): Liste af labels
    n_samples (int): Antal billeder per label
    seed (int): Seed for random number generator
    val_size (float): Andel af data, der skal bruges til validering
    test_size (float): Andel af data, der skal bruges til test
    batch_size (int): Batch size
    verbose (bool): Hvis True, printes status undervejs
    Returns:
    :
    Trænings-, validerings- og test-sæt
    """

    # sæt seed
    np.random.seed(seed)

    # hent doodles for hvert label
    X = []
    y = []
    for i, name in enumerate(tqdm(names, desc = f'Loading data...', disable = not verbose)):
        doodles = get_doodles(name, verbose = verbose)
        # normaliser doodles
        doodles = doodles / 255

        # transformer til [-1, 1]
        doodles = (doodles - 0.5) * 2

        # konkatener doodles og labels
        N_doodles = doodles.shape[0]
        # reshape doodles til (N_doodles, 28*28)
        X.append(doodles.reshape(N_doodles, 28*28))
        y.append(torch.ones(N_doodles) * i)

    # split i trænings-, og validerings-sæt (og måske test-sæt) med stratificering
    N = len(y)
    X = torch.cat(X, dim=0).unsqueeze(1)
    y = torch.cat(y, dim=0).unsqueeze(1)

    if test_size > 0:
        X, X_test, y, y_test = train_test_split(
            X, y,
            test_size = test_size,
            random_state = seed,
            stratify = y
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size = val_size,
        random_state = seed,
        stratify = y
    )

    # opret DataLoader objekter
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size = batch_size,
        sampler = RandomSampler(TensorDataset(X_train, y_train), num_samples = n_samples),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size = batch_size,
        sampler = RandomSampler(TensorDataset(X_val, y_val), num_samples = int(n_samples*val_size)),
    )
    if test_size > 0:
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size = batch_size,
            sampler = RandomSampler(TensorDataset(X_test, y_test), num_samples = int(n_samples*test_size)),
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader
    
def plot_doodles(labels: T.List[str], n_rows: int = 5, title: str = None) -> None:
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
        doodles = get_doodles(name, verbose = False)
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

def pca_doodles(n_components: int = 2, labels: T.List[str] = None, n_samples = 500) -> None:
    """
    PCA af doodles
    Args:
    n_components (int): Antal komponenter
    labels (List[str]): Liste af labels
    """
    if labels is None:
        labels = ['cat', 'dog', 'car', 'tree', 'house']

    # Hent doodles for hvert label
    X = []
    y = []

    for label, name in enumerate(labels):
        doodles = get_doodles(name, verbose = False)
        # normaliser doodles
        doodles = doodles / 255

        # transformer til [-1, 1]
        doodles = (doodles - 0.5) * 2

        # konkatener doodles og labels
        N_doodles = doodles.shape[0]
        # reshape doodles til (N_doodles, 28*28)
        X.append(doodles[:n_samples].reshape(n_samples, 28*28))
        y.append(np.full(n_samples, label))

    X = np.concatenate(X)
    y = np.concatenate(y)

    # PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if n_components == 2:
        # Plot PCA
        plt.figure(figsize=(8, 8))
        for i in range(len(labels)):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.5, label=labels[i])
        plt.title(f"PCA af doodles med {n_components} komponenter")
        plt.xlabel("Komponent 1")
        plt.ylabel("Komponent 2")
        plt.legend(loc='upper right', fontsize=8)

        plt.grid()
        plt.show()
    elif n_components == 3:
        # Interaktivt 3D plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(labels)):
            ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2], alpha=0.5, label=labels[i])
        ax.set_title(f"PCA af doodles med {n_components} komponenter")
        ax.set_xlabel("Komponent 1")
        ax.set_ylabel("Komponent 2")
        ax.set_zlabel("Komponent 3")
        ax.legend(loc='upper right', fontsize=8)

        ax.grid()
        plt.tight_layout()
        plt.show()

    else:
        print(f"Kan ikke plotte {n_components} komponenter. Vælg 2 eller 3 komponenter.")