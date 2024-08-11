import torch.nn as nn
from extras import *

class Generator(nn.Module):
    def __init__(self, noise_dim, max_length, vocab_size):
        super(Generator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.output_size = max_length * vocab_size
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.max_length, self.vocab_size)
        x = nn.Softmax(dim=2)(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class Conv_Discriminator(nn.Module):
    def __init__(self, max_length, vocab_size):
        super(Conv_Discriminator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        # input_size = max_length * vocab_size

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=vocab_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * max_length, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_length, self.vocab_size)  # Reshape to (batch_size, max_length, vocab_size)
        # print(x.shape)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, vocab_size, max_length)
        # print(x.shape)
        x = self.model(x)
        x = self.fc(x)
        return x