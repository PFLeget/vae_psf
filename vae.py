import numpy as np


class VAE:

    def __init__(self, **kwargs):
        raise NotImplementedError
    
    def initialize(self, data, **kwargs):
        raise NotImplementedError
    
    def train(self, **kwargs):
        raise NotImplementedError

    def encode(self, **kwargs):
        raise NotImplementedError
    
    def decode(self, **kwargs):
        raise NotImplementedError
    
    def predict(self, **kwargs):
        raise NotImplementedError
