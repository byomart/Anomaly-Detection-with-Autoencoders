import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, center_size):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.l3 = nn.Linear(hidden_size2, center_size)
        self.acti = nn.Tanh()

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.acti(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.acti(out)
        out = self.l3(out)
        return out
