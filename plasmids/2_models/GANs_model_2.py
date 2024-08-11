import torch.nn as nn
from extras import *

class Generator(nn.Module):
    def __init__(self, noise_dim, max_length, vocab_size):
        super(Generator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        feature_size = max_length * vocab_size
        self.model = nn.Sequential(
            nn.Linear(noise_dim, int(feature_size // 4)),
            nn.LeakyReLU(0.01),
            nn.Linear(int(feature_size // 4), int(feature_size // 2)),
            nn.LeakyReLU(0.01),
            nn.Linear(int(feature_size // 2), feature_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.max_length, self.vocab_size)
        return x 

class Discriminator(nn.Module):
    def __init__(self, max_length, vocab_size):
        super(Discriminator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        feature_size = max_length * vocab_size
        self.model = nn.Sequential(
            nn.Linear(feature_size, feature_size // 4),
            nn.LeakyReLU(0.01),
            nn.Linear(feature_size // 4, feature_size // 8),
            nn.LeakyReLU(0.01),
            nn.Linear(feature_size // 8, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.model(x)
        return x

# import torch.nn as nn
# from extras import *

# class Generator(nn.Module):
#     def __init__(self, noise_dim, max_length, vocab_size):
#         super(Generator, self).__init__()
#         self.max_length = max_length
#         self.vocab_size = vocab_size
#         feature_size = max_length * vocab_size
#         self.model = nn.Sequential(
#             nn.Linear(noise_dim, int(feature_size // 1.2)),
#             nn.LeakyReLU(0.01),
#             nn.Linear(int(feature_size // 1.2), int(feature_size // 1.1)),
#             nn.LeakyReLU(0.01),
#             nn.Linear(int(feature_size // 1.1), feature_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, self.max_length, self.vocab_size)
#         return x 

# class Discriminator(nn.Module):
#     def __init__(self, max_length, vocab_size):
#         super(Discriminator, self).__init__()
#         self.max_length = max_length
#         self.vocab_size = vocab_size
#         feature_size = max_length * vocab_size
#         self.model = nn.Sequential(
#             nn.Linear(feature_size, feature_size // 2),
#             nn.LeakyReLU(0.01),
#             nn.Linear(feature_size // 2, feature_size // 3),
#             nn.LeakyReLU(0.01),
#             nn.Linear(feature_size // 3, 1),
#             nn.Sigmoid()  
#         )

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         x = self.model(x)
#         return x