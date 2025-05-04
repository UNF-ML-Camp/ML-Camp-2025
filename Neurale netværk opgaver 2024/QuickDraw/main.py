from get_data import get_dataset
from data._data_static import TEGNINGER
from options import Hyperparameters, name_generator
from model import Net
from train import train

import random
import string

from torch.optim import Adam


# Sæt valgmuligheder
hyperparameters = Hyperparameters(
    lr = 0.001,
    optimizer = Adam,
    batch_size = 64,
)

# Hent data fra get_data.py
train_loader, val_loader = get_dataset(
    names=TEGNINGER,
    n_samples=30000,
    batch_size=hyperparameters.batch_size,
    verbose = True,
)

# Hent model architecturene fra model.py
model = Net(
    name = f"Efficient-CapsNet-lr={hyperparameters.lr}-{''.join(random.choices(string.ascii_lowercase,k=4))}", # name_generator(),
    hyperparameters=hyperparameters,
    n_capsules=16,
)

# tilføj optimizer til model
model.optimizer = model.hyperparameters.optimizer(
    model.parameters(),
    lr=model.hyperparameters.lr,
    betas=model.hyperparameters.betas,
    eps=model.hyperparameters.eps,
    weight_decay=model.hyperparameters.weight_decay,
)
setattr(model.hyperparameters, 'optimizer', model.optimizer.__class__.__name__)

# Træn modellen 
model_name = train(
    train_loader,
    val_loader,
    model,
)