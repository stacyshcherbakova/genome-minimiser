import torch
import torch.nn as nn
from genomes.extras import *

class LSTMGenerator(nn.Module):
    """
    This model takes noise as input, passes it through an LSTM network, and outputs a sequence 
    of probabilities for each position in the sequence using a Softmax layer.

    Attributes:
    -----------
    max_length - The length of the generated sequences.
    
    vocab_size - The size of the vocabulary for each position in the sequence.
    
    hidden_dim - The number of hidden units in each LSTM layer.
    
    num_layers - The number of LSTM layers in the generator.

    lstm - The LSTM network used for sequence generation.

    fc - A fully connected layer to transform LSTM outputs into sequence predictions.

    softmax - A softmax layer to normalize the output into probabilities for each position in the sequence.

    Methods:
    --------
    forward(x) - Forward pass through the generator, producing a sequence of probabilities.

    Parameters:
    -----------
    noise_dim - The dimensionality of the input noise vector.
    
    hidden_dim - The number of hidden units in each LSTM layer.
    
    num_layers - The number of LSTM layers.
    
    max_length - The maximum length of the generated sequence.
    
    vocab_size - The size of the vocabulary (e.g., number of categories for each position).

    """
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
    """
    This model takes generated or real sequences, passes them through an LSTM network, and outputs a binary 
    classification probability using a Sigmoid layer.

    Attributes:
    -----------
    max_length - The length of the input sequences.
    
    vocab_size - The size of the vocabulary for each position in the sequence.
    
    hidden_dim - The number of hidden units in each LSTM layer.
    
    num_layers - The number of LSTM layers in the discriminator.

    lstm - The LSTM network used for processing input sequences.

    fc - A fully connected layer to transform LSTM outputs into a binary classification.

    sigmoid - A Sigmoid activation layer to output a probability for binary classification.

    Methods:
    --------
    forward(x) - Forward pass through the discriminator, producing a probability that indicates if the input sequence is real or fake.

    Parameters:
    -----------
    hidden_dim - The number of hidden units in each LSTM layer.
    
    num_layers - The number of LSTM layers.
    
    max_length - The maximum length of the input sequence.
    
    vocab_size - The size of the vocabulary for each position in the sequence.
    """
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

class GRUGenerator(nn.Module):
    """
    This model takes noise as input, passes it through a GRU network, and outputs a sequence of probabilities 
    for each position in the sequence using a Softmax layer.

    Attributes:
    -----------
    max_length - The length of the generated sequences.
    
    vocab_size - The size of the vocabulary for each position in the sequence.
    
    hidden_dim - The number of hidden units in each GRU layer.
    
    num_layers - The number of GRU layers in the generator.

    gru - The GRU network used for sequence generation.

    fc - A fully connected layer to transform GRU outputs into sequence predictions.

    softmax - A softmax layer to normalize the output into probabilities for each position in the sequence.

    Methods:
    --------
    forward(x) - Forward pass through the generator, producing a sequence of probabilities.

    Parameters:
    -----------
    noise_dim - The dimensionality of the input noise vector.
    
    hidden_dim - The number of hidden units in each GRU layer.
    
    num_layers - The number of GRU layers.
    
    max_length - The maximum length of the generated sequence.
    
    vocab_size - The size of the vocabulary (e.g., number of categories for each position).
    """
    def __init__(self, noise_dim, hidden_dim, num_layers, max_length, vocab_size):
        super(GRUGenerator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, max_length * vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        out = out.view(-1, self.max_length, self.vocab_size)
        out = self.softmax(out)
        return out

class GRUDiscriminator(nn.Module):
    """
    This model takes generated or real sequences, passes them through a GRU network, and outputs a binary 
    classification probability using a Sigmoid layer.

    Attributes:
    -----------
    max_length - The length of the input sequences.
    
    vocab_size - The size of the vocabulary for each position in the sequence.
    
    hidden_dim - The number of hidden units in each GRU layer.
    
    num_layers - The number of GRU layers in the discriminator.

    gru - The GRU network used for processing input sequences.

    fc - A fully connected layer to transform GRU outputs into a binary classification.

    sigmoid - A Sigmoid activation layer to output a probability for binary classification.

    Methods:
    --------
    forward(x) - Forward pass through the discriminator, producing a probability that indicates if the input sequence is real or fake.

    Parameters:
    -----------
    hidden_dim - The number of hidden units in each GRU layer.
    
    num_layers - The number of GRU layers.
    
    max_length - The maximum length of the input sequence.
    
    vocab_size - The size of the vocabulary for each position in the sequence.
    """
    def __init__(self, hidden_dim, num_layers, max_length, vocab_size):
        super(GRUDiscriminator, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(vocab_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
