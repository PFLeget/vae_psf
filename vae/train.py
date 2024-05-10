import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import time
from .loss import VAELoss
from .models import VAE
import pickle

def load_train_data(data, train_ratio=0.8, batch_size=64, device='cpu'):
    """
    Load the training data and split it into training and validation sets.

    Parameters
    ----------
    data : numpy.ndarray
        The training data.
    train_ratio : float
        The ratio of the data to use for training.
    batch_size : int
        The batch size.
    device : str
        The device to use for training.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set.
    val_loader : torch.utils.data.DataLoader
        The data loader for the validation set.
    """
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32).to(device))

    # Split the dataset into training and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)

    return train_loader, val_loader


# Instantiate VAE model
# vae = VAE(N_pixel=28*28, layers_n_hidden_units=[512, 256, 128], latent_dim=2).to(device)

class training_loop:

    def __init__(self, vae_model, train_loader, val_loader, device='cpu', lr=1e-3, epochs=20):

        self.vae_model = vae_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.vae_loss = VAELoss().to(device)
        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=self.lr)
        self.train_size = len(self.train_loader.dataset)
        self.epochs = 20
        self.total_loss = []
        self.val_loss = []
    

def training_loop(vae_model, train_loader, val_loader, device='cpu'):


    epochs = 20
    total_loss = []
    val_loss = []

    TIMEA = time.time()

    print('start')
    for epoch in range(epochs):
        running_loss = 0.0
        for x, in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            x_hat, mu, log_var = vae(x)
            loss = vae_loss(x, x_hat, mu, log_var)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Add the mini-batch loss to the running loss
            running_loss += loss.item()

        for x_val, in val_loader:
            #x_hat_val, mu_val, log_var_val = vae(x_val)
            x_hat_val, mu_val, log_var_val = vae.forward(x_val, repam=False)
            vloss = vae_loss(x_val, x_hat_val, mu_val, log_var_val)
            val_loss.append(vloss.item())

        # Compute the average loss for the epoch
        epoch_loss = running_loss
        total_loss.append(epoch_loss)

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1} loss: {epoch_loss:.6f} validation loss: {vloss:.6f}")

TIMEB = time.time()

print(TIMEB-TIMEA)