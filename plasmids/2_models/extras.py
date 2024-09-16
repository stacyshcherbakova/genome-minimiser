import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset_Prep(Dataset):
    """
    This class is designed to take pre-encoded sequences (using a tokenizer), apply padding and 
    truncation as necessary to ensure that all sequences have a consistent length (max_len), and 
    return them in a format suitable for PyTorch models.

    Attributes:
    -----------
    encoded_sequences - A list of pre-encoded sequences, where each sequence is represented as a list of integers.
    
    max_len - The maximum sequence length. Sequences longer than this will be truncated, and those shorter 
    will be padded with zeros.
    
    padded_sequences - A tensor containing the padded and truncated sequences, ready for use in a PyTorch model.

    Methods:
    --------
    pad_and_truncate_sequences(sequences) - Pads or truncates the input sequences to a fixed 
    length (max_len) and converts them to a PyTorch tensor.

    __len__() - Returns the number of sequences in the dataset.

    __getitem__(idx) - Returns the sequence at the specified index from the padded sequences.

    """
    def __init__(self, encoded_sequences, max_len):
        self.max_len = max_len
        self.encoded_sequences = encoded_sequences
        self.padded_sequences = self.pad_and_truncate_sequences(self.encoded_sequences)

    def pad_and_truncate_sequences(self, sequences):
        processed_sequences = []
        for seq in sequences:
            if len(seq) > self.max_len:  # if more than max length then truncate
                seq = seq[:self.max_len]
            elif len(seq) < self.max_len:  # if less than max length then pad
                seq = list(seq) + [0] * (self.max_len - len(seq))
            processed_sequences.append(torch.tensor(seq, dtype=torch.long))
        return torch.stack(processed_sequences)

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, idx):
        return self.padded_sequences[idx]

class Base_Level_Tokenizer():
    """
    This tokenizer transforms character-level sequences into integer-encoded sequences and vice versa 
    using a fitted vocabulary. The vocabulary is built from the set of unique characters in the input sequences.

    Attributes:
    -----------
    le - A LabelEncoder instance to convert characters to integer indices and vice versa.
    
    vocab - A dictionary mapping each character to its corresponding integer index.

    Methods:
    --------
    fit_on_texts - Builds the vocabulary by fitting the LabelEncoder to the unique characters in the input sequences.

    sequence_to_indices - Converts character sequences into sequences of integer indices based on the fitted vocabulary.

    indices_to_sequence - Converts integer-encoded sequences back into character sequences using the inverse transform 
    of the LabelEncoder.

    """
    def __init__(self):
        self.le = LabelEncoder()
        self.vocab = {}

    def fit_on_texts(self, sequences: List[str]) -> Dict[str, int]:
        characters = list(set(''.join(sequences)))
        self.le.fit(characters)
        self.vocab = {str(char): int(index) for index, char in enumerate(self.le.classes_)}
        return self.vocab

    def sequence_to_indices(self, sequences: List[str]) -> List[List[int]]:
        return [list(map(int, self.le.transform(list(sequence)))) for sequence in sequences]

    def indices_to_sequence(self, sequences: List[List[int]]) -> List[str]:
        return [''.join(self.le.inverse_transform(list(map(int, sequence)))) for sequence in sequences]
    
class KMerTokenizer():
    """
    This tokenizer processes sequences by splitting them into non-overlapping k-mers and assigning each 
    k-mer a unique integer index based on the fitted vocabulary. It supports converting both directions: 
    from sequences to integer indices and vice versa.

    Attributes:
    -----------
    k - The length of the k-mers to generate from the sequences.
    
    vocab - A dictionary mapping each unique k-mer to its corresponding integer index.

    Methods:
    --------
    fit_on_texts - Builds the vocabulary by splitting the input sequences into k-mers and assigning a unique index 
    to each k-mer.

    sequence_to_indices - Converts sequences into lists of k-mer indices using the fitted vocabulary.

    indices_to_sequence - Converts lists of k-mer indices back into sequences of k-mers using the inverse of the vocabulary.
    """
    def __init__(self, k):
        self.k = k
        self.vocab = {}

    def fit_on_texts(self, sequences):
        k_mers = []
        for sequence in sequences:
            k_mers.extend([sequence[i:i+self.k] for i in range(0, len(sequence), self.k)])
        unique_k_mers = list(dict.fromkeys(k_mers)) 
        self.vocab = {k_mer: idx for idx, k_mer in enumerate(unique_k_mers)}
        return self.vocab

    def sequence_to_indices(self, sequences):
        return [[self.vocab[sequence[i:i+self.k]] for i in range(0, len(sequence), self.k)] for sequence in sequences]

    def indices_to_sequence(self, indices):
        inv_vocab = {idx: k_mer for k_mer, idx in self.vocab.items()}
        return [''.join([inv_vocab[idx] for idx in index_list]) for index_list in indices]

def train_generator(fake_data, discriminator, optimizer_g, criterion):
    """
    This function computes the loss for the generator by passing fake data through the 
    discriminator, calculates the gradient of the loss, and updates the generator's weights.

    Parameters:
    -----------
    fake_data - The generated (fake) data from the generator.
    
    discriminator - The discriminator model that classifies the fake data.
    
    optimizer_g - The optimizer for updating the generator's parameters.
    
    criterion - The loss function to evaluate the generator's performance (e.g., Binary Cross-Entropy).

    Returns:
    --------
    g_loss.item() - The loss value of the generator after the update.

    """

    optimizer_g.zero_grad()

    labels = torch.ones(fake_data.size(0), 1).to(device)
    outputs = discriminator(fake_data)
    g_loss = criterion(outputs, labels)

    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()

def train_generator_2(fake_data, critic, optimizer_g):
    """
    This function computes the loss for the generator using the Wasserstein loss, 
    updates the generator's parameters, and minimizes the critic's score for the fake data.

    Parameters:
    -----------
    fake_data - The generated (fake) data from the generator.
    
    critic - The critic model (analogous to a discriminator in standard GANs) used for evaluating fake data.
    
    optimizer_g - The optimizer for updating the generator's parameters.

    Returns:
    --------
    g_loss.item() - The Wasserstein loss value for the generator after the update.

    """
    optimizer_g.zero_grad()

    outputs = critic(fake_data)

    g_loss = -torch.mean(outputs)

    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()

def train_critic(real_data, fake_data, optimizer_d, critic, clip_value):
    """
    The function computes the Wasserstein loss for the critic by calculating the difference 
    between real and fake outputs. It applies weight clipping to maintain the Lipschitz constraint.

    Parameters:
    -----------
    real_data - The real data from the dataset.
    
    fake_data - The generated (fake) data from the generator.
    
    optimizer_d - The optimizer for updating the critic's parameters.
    
    critic - The critic model that scores real and fake data.
    
    clip_value - The value used for weight clipping to ensure the Lipschitz constraint.

    Returns:
    --------
    d_loss.item() - The Wasserstein loss for the critic after the update.
    
    real_real_score - The accuracy of the critic in classifying real data as real.
    
    fake_real_score - The accuracy of the critic in classifying fake data as fake.
    """
    optimizer_d.zero_grad()

    real_outputs = critic(real_data)
    fake_outputs = critic(fake_data.detach())

    d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)
    d_loss.backward()
    optimizer_d.step()

    for p in critic.parameters():
        p.data.clamp_(-clip_value, clip_value)

    # Calculate accuracy
    real_real_score = torch.mean((real_outputs > 0).float()).item()
    fake_real_score = torch.mean((fake_outputs < 0).float()).item()

    return d_loss.item(), real_real_score, fake_real_score


def train_discriminator(real_data, fake_data, discriminator, optimizer_d, criterion):
    """
    The function computes the loss for the discriminator by evaluating real and fake data, 
    then backpropagates the loss and updates the discriminator's weights.

    Parameters:
    -----------
    real_data - The real data from the dataset.
    
    fake_data - The generated (fake) data from the generator.
    
    discriminator - The discriminator model that classifies real and fake data.
    
    optimizer_d - The optimizer for updating the discriminator's parameters.
    
    criterion - The loss function to evaluate the discriminator's performance (e.g., Binary Cross-Entropy).

    Returns:
    --------
    d_loss.item() - The combined loss for real and fake data after the update.
    
    real_accuracy - The accuracy of the discriminator in classifying real data as real.
    
    fake_accuracy - The accuracy of the discriminator in classifying fake data as fake.

    """
        
    optimizer_d.zero_grad()

    real_labels = torch.ones(real_data.size(0), 1).to(device)
    fake_labels = torch.zeros(fake_data.size(0), 1).to(device)

    real_outputs = discriminator(real_data)
    d_loss_real = criterion(real_outputs, real_labels)
    real_predictions = (real_outputs >= 0.5).float()
    real_accuracy = (real_predictions == real_labels).float().mean().item()

    fake_outputs = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_outputs, fake_labels)
    fake_predictions = (fake_outputs < 0.5).float()
    fake_accuracy = (fake_predictions == fake_labels).float().mean().item()

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()

    return d_loss.item(), real_accuracy, fake_accuracy

def one_hot_encode_sequences(sequences, vocab_size):
    """
    Given a batch of sequences and a vocabulary size, this function produces a one-hot 
    encoded tensor for each sequence, where each index is represented by a binary vector.

    Parameters:
    -----------
    sequences - A tensor of shape (batch_size, sequence_length) containing integer-encoded sequences.
    
    vocab_size - The size of the vocabulary (number of unique indices).

    Returns:
    --------
    one_hot_encoded - A tensor of shape (batch_size, sequence_length, vocab_size) where each index is 
    represented by a one-hot encoded vector.
    
    Raises:
    -------
    ValueError - If any index in the sequences exceeds the vocabulary size.

    """
        
    if sequences.max() >= vocab_size:
        raise ValueError(f"Found index {sequences.max()} in sequences which is out of bounds for vocab_size {vocab_size}")

    batch_size = sequences.size(0)
    max_length = sequences.size(1)
    # print(f"Batch size: {batch_size}, Max length: {max_length}, Vocab size: {vocab_size}")
    one_hot_encoded = torch.zeros(batch_size, max_length, vocab_size).to(sequences.device)
    one_hot_encoded.scatter_(2, sequences.unsqueeze(2), 1)
    # print(f"One-hot encoded shape: {one_hot_encoded.shape}")
    return one_hot_encoded

def smooth_labels(labels, smoothing=0.1):
    """
    Label smoothing helps to regularize the model by softening the labels, 
    reducing the model's confidence on exact label predictions.

    Parameters:
    -----------
    labels - The original binary or categorical labels.
    
    smoothing - The amount of smoothing to apply. Default is 0.1.

    Returns:
    --------
    labels - The smoothed labels with a mixture of original labels and uniform distribution.

    """
    with torch.no_grad():
        labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    return labels

def perform_pca(real_data, fake_data, n_components=4):
    """
    This function reduces the dimensionality of both real and fake data for visualization or 
    analysis purposes, keeping the top components as specified by `n_components`.

    Parameters:
    -----------
    real_data - The real data from the dataset, typically before PCA.
    
    fake_data - The fake (generated) data from the generator, typically before PCA.
    
    n_components - The number of principal components to keep. Default is 4.

    Returns:
    --------
    real_pca - The PCA-transformed real data.
    
    fake_pca - The PCA-transformed fake data.

    """
    pca = PCA(n_components=n_components)
    real_data_np = real_data.cpu().numpy()
    fake_data_np = fake_data.cpu().numpy()

    # print(f'real_data_np shape: {real_data_np.shape}') 
    # print(f'fake_data_np shape: {fake_data_np.shape}')  

    real_pca = pca.fit_transform(real_data_np)
    fake_pca = pca.transform(fake_data_np)

    return real_pca, fake_pca

def plot_pca(real_pca, fake_pca):
    """
    This function visualizes the PCA-reduced real and fake data, plotting the 
    first two principal components to highlight differences between the two datasets.

    Parameters:
    -----------
    real_pca - The PCA-transformed real data.
    
    fake_pca - The PCA-transformed fake data.

    Returns:
    --------
    None

    """
    plt.figure(figsize=(10, 5))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], label='Real Data', alpha=0.6)
    plt.scatter(fake_pca[:, 0], fake_pca[:, 1], label='Fake Data', alpha=0.6)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA of Real vs Fake Data')
    plt.legend()
    plt.show()

def gradient_penalty(critic, real_data, fake_data):
    """
    This function ensures the Lipschitz continuity of the critic by penalizing the 
    gradient norm between real and fake data, which helps stabilize the training of WGAN-GP.

    Parameters:
    -----------
    critic - The critic model in the WGAN-GP framework.
    
    real_data - The real data from the dataset.
    
    fake_data - The generated (fake) data from the generator.

    Returns:
    --------
    penalty - The gradient penalty, calculated as the mean squared error of the gradient norms.
    
    """
    batch_size, _, _ = real_data.shape
    epsilon = torch.rand(batch_size, 1, 1, device=real_data.device)
    epsilon = epsilon.expand_as(real_data)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data

    interpolated.requires_grad_(True)
    interpolated_critic = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=interpolated_critic,
        inputs=interpolated,
        grad_outputs=torch.ones(interpolated_critic.size(), device=real_data.device),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty