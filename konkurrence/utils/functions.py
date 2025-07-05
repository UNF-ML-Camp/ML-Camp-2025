import numpy as np
import typing as T
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
import random

SEED = 42

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

def linear_schedule(beta_start, beta_end, num_timesteps):
  betas = torch.linspace(beta_start, beta_end, num_timesteps)
  return betas

def cosine_schedule(num_timesteps, s=0.008):
  def f(t):
    return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
  x = torch.linspace(0, num_timesteps, num_timesteps + 1)
  alphas_cumprod = f(x) / f(torch.tensor([0]))
  betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
  betas = torch.clip(betas, 0.0001, 0.999)
  return betas

def square_root_schedule(max_timesteps, s=8e-3):
    def f(t, T):
        return 1 - torch.sqrt((t / T) + s)
    x = torch.linspace(0, max_timesteps, max_timesteps + 1)
    alpha_bar_t = f(x, max_timesteps)
    alpha_bar_t_1 = f(x[1:], max_timesteps)
    alpha_cumprod = alpha_bar_t_1 / alpha_bar_t[:-1]
    betas = 1 - alpha_cumprod
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas

def visualize_schedulers(dataset):
  image = next(iter(dataset))[0]

  beta_linear = linear_schedule(1e-4, 2e-2, 1001)
  beta_cosine = cosine_schedule(1001)
  beta_sqr = square_root_schedule(1001)
  alpha_linear = 1 - beta_linear
  alpha_linear_prod = torch.cumprod(alpha_linear, dim=0)

  alpha_cosine = 1 - beta_cosine
  alpha_cosine_prod = torch.cumprod(alpha_cosine, dim=0)

  alpha_sqr = 1 - beta_sqr
  alpha_sqr_prod = torch.cumprod(alpha_sqr, dim=0)

  image = image * 0.5 + 0.5  # Normalize to [-1, 1]

  epsilon = torch.randn_like(image)

  ts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  images = []

  for t in ts:
      image_linear_noisy = (alpha_linear_prod[t] * image + np.sqrt(1 - alpha_linear_prod[t]) * epsilon).reshape(3, 16, 16)
      image_linear_noisy = image_linear_noisy.permute(1, 2, 0)  # Swap axes for correct RGB order
      image_linear_noisy = torch.clamp(image_linear_noisy, 0.0, 1.0)
      images.append(image_linear_noisy.cpu().numpy())

      image_cosine_noisy = (alpha_cosine_prod[t] * image + np.sqrt(1 - alpha_cosine_prod[t]) * epsilon).reshape(3, 16, 16)
      image_cosine_noisy = image_cosine_noisy.permute(1, 2, 0)  # Swap axes for correct RGB order
      image_cosine_noisy = torch.clamp(image_cosine_noisy, 0.0, 1.0)
      images.append(image_cosine_noisy.cpu().numpy())

      image_sqr_noisy = (alpha_sqr_prod[t] * image + np.sqrt(1 - alpha_sqr_prod[t]) * epsilon).reshape(3, 16, 16)
      image_sqr_noisy = image_sqr_noisy.permute(1, 2, 0)  # Swap axes for correct RGB order
      image_sqr_noisy = torch.clamp(image_sqr_noisy, 0.0, 1.0)
      images.append(image_sqr_noisy.cpu().numpy())

  images = np.array(images)


  fig, axs = plt.subplots(3, len(ts), figsize=(2 * len(ts), len(ts)//2))
  for i, t in enumerate(ts):
      axs[0, i].imshow(images[i * 3])
      axs[0, i].set_title(f"t={t}", fontsize=14)
      axs[1, i].imshow(images[i * 3 + 1])
      axs[2, i].imshow(images[i * 3 + 2])
      
      if i == 0:
          axs[0, i].set_ylabel("Method 1", fontsize=14)
          axs[1, i].set_ylabel("Method 2", fontsize=14)
          axs[2, i].set_ylabel("Method 3", fontsize=14)

          # Remove ticks and square axes but keep the labels
          axs[0, i].set_xticks([])
          axs[0, i].set_yticks([])
          axs[1, i].set_xticks([])
          axs[1, i].set_yticks([])
          axs[2, i].set_xticks([])
          axs[2, i].set_yticks([])

      else:
          axs[0, i].set_xticks([])
          axs[0, i].set_yticks([])
          axs[1, i].set_xticks([])
          axs[1, i].set_yticks([])
          axs[2, i].set_xticks([])
          axs[2, i].set_yticks([])

  plt.tight_layout()
  plt.show()

def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i], fontweight='bold')

    if title is not None:
        plt.suptitle(title, fontweight='bold')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_sprites(labels, dataset=None, n_rows: int = 5, title: str = None):
    """
    Plot sprites grouped by label.

    Args:
    labels (List[str]): List of class names.
    dataset (Dataset): Dataset to draw samples from. Must return (image, label).
    n_rows (int): Total number of rows/columns for the image grid.
    title (str): Title of the plot.
    """
    assert dataset is not None, "You must provide a dataset to sample from."

    N = n_rows ** 2
    N_per_label = max(1, N // len(labels))

    images = []
    targets = []

    # Collect N_per_label samples per class
    collected = {i: [] for i in range(len(labels))}

    for img, label in dataset:
        label = label.item() if torch.is_tensor(label) else label
        if len(collected[label]) < N_per_label:
            collected[label].append(img)
        if all(len(v) == N_per_label for v in collected.values()):
            break

    # Format the images for display
    for label_idx, class_imgs in collected.items():
        for img in class_imgs:
            img = img.detach().cpu()  # Ensure it's on CPU
            img = img * 0.5 + 0.5      # Unnormalize from [-1, 1] → [0, 1]
            if img.shape[0] == 1:
                img = img.squeeze(0).numpy()  # Grayscale (C=1) → (H, W)
            elif img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()  # RGB (C, H, W) → (H, W, C)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            images.append(img)
            targets.append(label_idx)

    # Plotting
    fig, axs = plt.subplots(n_rows, n_rows, figsize=(n_rows, n_rows), constrained_layout=True)
    for i, (img, label_idx) in enumerate(zip(images, targets)):
        r, c = i // n_rows, i % n_rows
        axs[r, c].imshow(img, cmap="gray")
        axs[r, c].set_title(f"{labels[label_idx]}", fontsize=8)
        axs[r, c].axis("off")

    if title is None:
        title = f"{N} sprites from SpritesDataset"
    plt.suptitle(title)
    plt.show()

def pca_sprites(
    n_components: int = 2,
    class_names:  list = None,
    n_samples:    int = 500,
    sprite_path:  str = "./data/sprites.npy",
    label_path:   str = "./data/labels.npy",
    random_state: int = 0
) -> None:
    """
    PCA af sprites
    Args:
    n_components (int): Antal komponenter
    labels (List[str]): Liste af labels
    """

    set_seed()  # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)
    sprites = np.load(sprite_path,  allow_pickle=True)    # (N, 3, 16, 16)
    labels_raw  = np.load(label_path, allow_pickle=True)      # (N, 5)
    labels = labels_raw.argmax(axis=1)

    # make sure we handle both channel orders
    if sprites.shape[-1] == 3:
        sprites = sprites.transpose(0, 3, 1, 2)

    if class_names is None:
        class_names = np.unique(labels)

    X_list, y_list = [], []

    for cls in class_names:
        idx = np.where(labels == cls)[0]
        
        chosen = rng.choice(idx,
                            size=min(n_samples, idx.size),
                            replace=False)
        
        sprites_cls = sprites[chosen].astype(np.float32) / 255.0
        sprites_cls = sprites_cls.reshape(sprites_cls.shape[0], -1)

        X_list.append(sprites_cls)
        y_list.append(np.full(sprites_cls.shape[0], cls, dtype=labels.dtype))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    return X_pca, y, np.array(class_names, dtype=object)