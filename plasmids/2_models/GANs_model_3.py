import torch
import torch.nn as nn
from extras import *

class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, num_layers, max_length, vocab_size):
        super(LSTMGenerator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, max_length * vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # print("Generator input shape", x.shape)
        # print(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # print("Generator input shape after unsqueeze", x.shape)
        # print(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # print(h0.shape)
        # print(c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        # print("LSTM output shape: ", out.shape)
        out = self.fc(out[:, -1, :])
        out = out.view(-1, self.max_length, self.vocab_size)
        out = self.softmax(out)
        return out

class LSTMDiscriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers, max_length, vocab_size):
        super(LSTMDiscriminator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("Discriminator input shape: ", x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # print(h0.shape)
        # print(c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        # print("LSTM output shape: ", out.shape)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out