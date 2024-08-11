import torch.nn as nn
from extras import *

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, int(noise_dim // 1.05)),
            nn.BatchNorm1d(int(noise_dim // 1.05)),
            nn.LeakyReLU(0.01),
            nn.Linear(int(noise_dim // 1.05), int(noise_dim // 1.12)),
            nn.BatchNorm1d(int(noise_dim // 1.12)),
            nn.LeakyReLU(0.01),
            nn.Linear(int(noise_dim // 1.12), output_dim)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, int(input_dim // 2)),
            nn.LeakyReLU(0.01),
            nn.Linear(int(input_dim // 2), int(input_dim // 3)),
            nn.LeakyReLU(0.01),
            nn.Linear(int(input_dim // 3), 1)
        )

    def forward(self, x):
        x = self.main(x)
        return x