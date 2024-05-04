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


train_ratio = 0.8
batch_size = 64
device = 'cpu'
# device = 'mps'

dataset = TensorDataset(torch.tensor(normed_Z, dtype=torch.float32).to(device))

# Split the dataset into training and validation sets
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)


# Instantiate VAE model
vae = VAE(N_pixel=28*28, layers_n_hidden_units=[512, 256, 128], latent_dim=2).to(device)

# Define MSE loss function
vae_loss = VAELoss().to(device)

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

train_size = len(train_loader.dataset)
print(train_size)

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