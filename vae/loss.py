import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, x, x_hat, mu, sigma):
        """
        TO DO THE DOC.
        """
        # Reconstruction loss (MSE)
        mse_loss = torch.sum((x - x_hat) ** 2)

        # KL divergence loss
        # kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        # Total loss
        total_loss = mse_loss + kl_loss
        return total_loss