"""Dataclasses til hyperparameters, training, og data valgmuligheder"""
import random
import string
from datetime import datetime
import typing as T
from dataclasses import dataclass, asdict
from torch import optim    

@dataclass
class Hyperparameters:
    # data parameters
    batch_size: T.Union[int, T.List[int]] = 32

    # training parametre
    epochs: T.Union[int, T.List[int]] = 10

    # optimizer parametre
    optimizer: optim.Optimizer = optim.SGD
    lr: T.Union[float, T.List[float]] = 0.001
    betas: T.Tuple[float] = (0.9, 0.999)
    dampening: float = 0
    eps: float = 1e-08,
    momentum: T.Union[float, T.List[float]] = 0.5
    nesterov: bool = False
    weight_decay: float = 0

def name_generator():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + ''.join(random.choices(string.ascii_lowercase,k=4))

def get_hyperparameters(hyperparameters: Hyperparameters, optimizer: optim.Optimizer):
    """
    Hent hyperparametre der er relevante for modellen at logge
    Args:
    hyperparameters (Hyperparameters): Hyperparametre
    optimizer (optim.Optimizer): Optimizer
    Returns:
    dict: Hyperparametre
    """
    hyperparams_dict = asdict(hyperparameters)
    optimizer_args = optimizer.defaults

    # find fællesmængden af hyperparametre vi definerer i Hyperparameters og dem der bruges i den valgte optimizer
    optimizer_params = set(hyperparams_dict.keys()).intersection(
        set(optimizer_args.keys()))
    
    # Tilføj batch_size og epochs
    optimizer_params.update(['batch_size', 'epochs', 'optimizer'])

    # Omdan til dict med værdierne
    hyperparameters = {
        key: hyperparams_dict[key] for key in optimizer_params}
    
    return hyperparameters