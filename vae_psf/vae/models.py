import torch
import torch.nn as nn
import copy
import numpy as np

def size_conv_out(in_pixel, kernel_size, padding=0., stride=1.):
    out_pixel = (in_pixel + 2.*padding - kernel_size) / stride
    out_pixel += 1.
    return int(np.floor(out_pixel))

def size_convt_out(in_pixel, kernel_size, padding=0, stride=1.):
    out_pixel = (in_pixel-1)*stride + kernel_size -2*padding
    return out_pixel


class VAE(nn.Module):
    def __init__(self, N_pixel=16*16, layers_n_hidden_units=[128, 64, 32], latent_dim=4):
        super(VAE, self).__init__()

        self._npixel = N_pixel
        self._layers_n_hidden_units_encode = copy.deepcopy(layers_n_hidden_units)
        layers_n_hidden_units.reverse()
        self._layers_n_hidden_units_decode = layers_n_hidden_units
        self._latent_dim = latent_dim

        # Encoder
        self.model_encode = nn.Sequential(
            nn.Linear(self._npixel, self._layers_n_hidden_units_encode[0]))
        I = 0
        I += 1
        self.model_encode.add_module(str(I), nn.ReLU())

        n_hidden_layer = len(layers_n_hidden_units) - 1

        for hidden_layer in range(n_hidden_layer):
            I += 1
            self.model_encode.add_module(str(I), nn.Linear(self._layers_n_hidden_units_encode[hidden_layer],
                                                           self._layers_n_hidden_units_encode[hidden_layer+1],
                                                    ))
            I += 1
            self.model_encode.add_module(str(I), nn.ReLU())

        print(self.model_encode)

        # VAE reparameterization
        self.fc_mu = nn.Linear(self._layers_n_hidden_units_encode[-1], self._latent_dim)
        self.fc_logvar = nn.Linear(self._layers_n_hidden_units_encode[-1], self._latent_dim)

        # Decoder
        self.model_decode = nn.Sequential(
            nn.Linear(self._latent_dim, self._layers_n_hidden_units_decode[0]))
        I = 0
        I += 1
        self.model_decode.add_module(str(I), nn.ReLU())

        for hidden_layer in range(n_hidden_layer):
            I += 1
            self.model_decode.add_module(str(I), nn.Linear(self._layers_n_hidden_units_decode[hidden_layer],
                                                           self._layers_n_hidden_units_decode[hidden_layer+1],
                                                           ))
            I += 1
            self.model_decode.add_module(str(I), nn.ReLU())

        I += 1
        self.model_decode.add_module(str(I), nn.Linear(self._layers_n_hidden_units_decode[-1], self._npixel))
        # I += 1
        # self.model_decode.add_module(str(I), nn.Sigmoid())
        print(self.model_decode)
        self.N = torch.distributions.Normal(0, 1)

    def encoder(self, x):
        h = self.model_encode(x)
        return h

    def decoder(self, z):
        x_hat = self.model_decode(z)
        return x_hat

    def reparameterize(self, mu, sigma):
        eps = self.N.sample(mu.shape)
        return mu + eps*sigma

    def forward(self, x, repam=True):
        h1 = self.encoder(x)
        mu = self.fc_mu(h1)
        sigma = torch.exp(self.fc_logvar(h1))
        if repam:
            z = self.reparameterize(mu, sigma)
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(mu)
        return x_hat, mu, sigma


class ConvVAE(nn.Module):
    def __init__(self,
                 n_pixel=16,
                 in_channel=1,
                 layer_n_channels=[32, 64, 128, 256],
                 kernel_sizes=[3, 3, 3, 3],
                 strides=[1, 1, 1, 1],
                 paddings=[0, 0, 0, 0],
                 latent_dim=4,
                 activation=nn.ReLU(),
                 device='cpu',
                 ):
        super(ConvVAE, self).__init__()

        self._n_pixel = n_pixel
        self._n_pixel_out = n_pixel
        self._latent_dim = latent_dim
        self._activation = activation
        self._in_channel = in_channel
        self._layer_n_channels = copy.deepcopy(layer_n_channels)
        self._kernel_sizes = copy.deepcopy(kernel_sizes)
        self._strides = copy.deepcopy(strides)
        self._paddings = copy.deepcopy(paddings)
        self._n_layers = len(layer_n_channels)

        struct_in = [self._n_pixel_out]
        for i in range(self._n_layers):
            self._n_pixel_out = size_conv_out(self._n_pixel_out,
                                              kernel_sizes[i],
                                              padding=paddings[i],
                                              stride=strides[i])
            struct_in.append(self._n_pixel_out)


        # Encoder
        self.model_encode = nn.Sequential(
            nn.Conv2d(self._in_channel,
                      layer_n_channels[0],
                      kernel_sizes[0],
                      stride=strides[0],
                      padding=paddings[0])
        )
        I = 0
        I += 1
        self.model_encode.add_module(str(I), activation)

        n_layer = len(self._layer_n_channels) - 1

        for layer in range(n_layer):
            I += 1
            self.model_encode.add_module(str(I),
                                         nn.Conv2d(layer_n_channels[layer],
                                                   layer_n_channels[layer + 1],
                                                   kernel_sizes[layer + 1],
                                                   stride=strides[layer + 1],
                                                   padding=paddings[layer + 1]),
                                         )
            I += 1
            self.model_encode.add_module(str(I), activation)
        I += 1
        self.model_encode.add_module(str(I), nn.Flatten())

        print(self.model_encode)

        # VAE re-parameterization
        self.fc_mu = nn.Linear(self._n_pixel_out**2 * layer_n_channels[-1], self._latent_dim)
        self.fc_log_sigma = nn.Linear(self._n_pixel_out**2 * layer_n_channels[-1], self._latent_dim)

        # Decoder

        layer_n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()

        n_pixel_out = copy.deepcopy(self._n_pixel_out)
        struct_out = [n_pixel_out]
        for i in range(self._n_layers):
            n_pixel_out = size_convt_out(n_pixel_out,
                                         kernel_sizes[i],
                                         padding=paddings[i],
                                         stride=strides[i])
            struct_out.append(n_pixel_out)

        self.model_decode = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_pixel_out**2 * layer_n_channels[0]),
            nn.Unflatten(1, (layer_n_channels[0], self._n_pixel_out, self._n_pixel_out)),
        )
        I = 1

        for layer in range(n_layer):
            I += 1
            self.model_decode.add_module(str(I),
                                 nn.ConvTranspose2d(layer_n_channels[layer],
                                                    layer_n_channels[layer + 1],
                                                    kernel_sizes[layer + 1],
                                                    stride=strides[layer + 1],
                                                    padding=paddings[layer + 1]),
                                         )
            I += 1
            self.model_decode.add_module(str(I), activation)

        I += 1
        self.model_decode.add_module(str(I),
                                     nn.ConvTranspose2d(layer_n_channels[-1],
                                                        self._in_channel,
                                                        kernel_sizes[-1],
                                                        stride=strides[-1],
                                                        padding=paddings[-1]),
                                     )
        print(self.model_decode)

        print(struct_in)
        print(struct_out)

        mean = torch.tensor(0.0).to(device)
        std = torch.tensor(1.0).to(device)
        self.N = torch.distributions.Normal(mean, std)

    def encoder(self, x):
        h = self.model_encode(x)
        return h

    def decoder(self, z):
        x_hat = self.model_decode(z)
        return x_hat

    def reparameterize(self, mu, sigma):
        eps = self.N.sample(mu.shape)
        return mu + eps * sigma

    def forward(self, x, repam=True):
        h1 = self.encoder(x)
        mu = self.fc_mu(h1)
        sigma = torch.exp(self.fc_log_sigma(h1))
        if repam:
            z = self.reparameterize(mu, sigma)
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(mu)
        return x_hat, mu, sigma