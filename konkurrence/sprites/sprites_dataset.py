import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import numpy as np

SEED = 1
CLASS_LABELS = ['human', 'non-human', 'food', 'spell', 'side-facing']
TRAIN_SIZE = 40000
VAL_SIZE = 10000
TEST_SIZE = 1000
DATASET_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

class SpritesDataset(Dataset):
    def __init__(self, 
            transform,
            sfilename='./data/sprites.npy', 
            lfilename='./data/labels.npy', 
            num_samples=40000,
            seed=1,
        ):
        self.images = np.load(sfilename, allow_pickle=True)
        labels = np.load(lfilename, allow_pickle=True)
        self.labels = np.argmax(labels, axis=1) 

        # Reduce dataset size
        if num_samples:
            sampled_indeces = random.sample(range(len(self.images)), num_samples)
            self.images = self.images[sampled_indeces]
            self.labels = self.labels[sampled_indeces]


        self.transform = transform
       
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def prepare_dataloaders(batch_size=100, val_batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])
    
    dataset = SpritesDataset(transform, num_samples=DATASET_SIZE, seed=SEED)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, VAL_SIZE, TEST_SIZE], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader