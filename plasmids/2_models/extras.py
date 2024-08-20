import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset_Prep(Dataset):
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
    optimizer_g.zero_grad()

    labels = torch.ones(fake_data.size(0), 1).to(device)
    outputs = discriminator(fake_data)
    g_loss = criterion(outputs, labels)

    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()

def train_generator_2(fake_data, critic, optimizer_g):
    optimizer_g.zero_grad()

    outputs = critic(fake_data)

    g_loss = -torch.mean(outputs)

    g_loss.backward()
    optimizer_g.step()

    return g_loss.item()

def train_critic(real_data, fake_data, optimizer_d, critic, clip_value):
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
    with torch.no_grad():
        labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    return labels

def perform_pca(real_data, fake_data, n_components=4):
    pca = PCA(n_components=n_components)
    real_data_np = real_data.cpu().numpy()
    fake_data_np = fake_data.cpu().numpy()

    # print(f'real_data_np shape: {real_data_np.shape}') 
    # print(f'fake_data_np shape: {fake_data_np.shape}')  

    real_pca = pca.fit_transform(real_data_np)
    fake_pca = pca.transform(fake_data_np)

    return real_pca, fake_pca

def plot_pca(real_pca, fake_pca):
    plt.figure(figsize=(10, 5))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], label='Real Data', alpha=0.6)
    plt.scatter(fake_pca[:, 0], fake_pca[:, 1], label='Fake Data', alpha=0.6)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA of Real vs Fake Data')
    plt.legend()
    plt.show()

def gradient_penalty(critic, real_data, fake_data):
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

class Dataset_Prep(Dataset):
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
    