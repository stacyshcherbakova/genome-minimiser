import torch.nn as nn
from genomes.extras import *

class Generator(nn.Module):
    """
    This model takes random noise as input and produces a sequence of softmax-activated 
    outputs corresponding to the vocabulary size. It is useful for generating discrete sequences 
    such as DNA or protein sequences.

    Attributes:
    -----------
    max_length - The length of the generated sequences.
    
    vocab_size - The size of the vocabulary for each position in the sequence (e.g., number of unique bases or amino acids).
    
    output_size - The total size of the output sequence, calculated as `max_length * vocab_size`.

    model - The neural network layers, consisting of three fully connected (Linear) layers with ReLU activations.

    Methods:
    --------
    forward(x) - Forward pass through the generator, producing a sequence of probabilities for each 
    position in the sequence, activated by Softmax.

    Parameters:
    -----------
    noise_dim - The dimensionality of the input noise vector used to generate the sequence.
    
    max_length - The maximum length of the generated sequence.
    
    vocab_size - The size of the vocabulary (e.g., number of possible categories for each position).
     
    """
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
    """
    This model takes input sequences and classifies them using several fully connected layers 
    with LeakyReLU activations and dropout for regularization.

    Attributes:
    -----------
    model - The neural network layers, consisting of four fully connected (Linear) layers with LeakyReLU activations, 
    dropout for regularization, and a final Sigmoid activation for binary classification.

    Methods:
    --------
    forward(x) - Forward pass through the discriminator, producing a binary classification output for each input sequence.

    Parameters:
    -----------
    input_size - The size of the input sequence (i.e., max_length * vocab_size).

    """
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class Critic(nn.Module):
    """
    The critic uses a similar structure to the discriminator but without the final Sigmoid activation.
    It outputs a score indicating the "realness" of the input sequence, where higher scores 
    indicate more realistic sequences.

    Attributes:
    -----------
    model - The neural network layers, consisting of fully connected (Linear) layers with LeakyReLU activations 
    and dropout for regularization.

    Methods:
    --------
    forward(x) - Forward pass through the critic, producing a real-valued score for each input sequence.

    Parameters:
    -----------
    input_size - The size of the input sequence (i.e., max_length * vocab_size).

    """
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Conv_Discriminator(nn.Module):
    """
    This model takes sequence data, applies convolutional layers to extract features, 
    and classifies the sequence using fully connected layers. It's useful for tasks where 
    spatial relationships between elements in the sequence are important.

    Attributes:
    -----------
    max_length - The length of the input sequences.
    
    vocab_size - The size of the vocabulary (e.g., number of unique bases or amino acids in the sequences).

    model - The sequence of 1D convolutional layers with LeakyReLU activations.

    fc - The fully connected layers after the convolutional layers, ending with a Sigmoid activation for binary classification.

    Methods:
    --------
    forward(x) - Forward pass through the convolutional discriminator, producing a binary classification 
        output for each input sequence.

    Parameters:
    -----------
    max_length - The length of the input sequences.
    
    vocab_size - The size of the vocabulary for each position in the sequence (e.g., number of unique categories for each position).

    """
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