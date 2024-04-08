import torch
import torch.nn as nn
from classes.encoder import Encoder
from classes.decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, center_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size1, hidden_size2, center_size)
        self.decoder = Decoder(output_size, hidden_size1, hidden_size2, center_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
