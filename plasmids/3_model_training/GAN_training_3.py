import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_models')))

from GANs_model_3 import *
from GAN_training_2 import train
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Using LSTM (long short term memroy) generato rna discriminator which tak einto account the previous data in the seqience - good for sequenctial data, was suggested by chat GPT

def main():
    # Parameters
    sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrappingcleaned_sequences.npy', allow_pickle=True) # ["ATCGTACG", "CGTACGTAGC", "ATCG", "CGTACGTAGCTAGCGT"] 
    max_length = 6000 
    batch_size = 2
    num_epochs = 1000 
    noise_dim = 600
    hidden_dim = 128
    num_layers = 2
    fig_name = "lstm"

    # Tokenization and dataset preparation
    base_tokenizer = Base_Level_Tokenizer()
    base_vocab = base_tokenizer.fit_on_texts(sequences)
    vocab_size = len(base_vocab)
    print("Base Tokenizer Vocabulary:", base_vocab)

    base_sequence_encoded = base_tokenizer.sequence_to_indices(sequences)
    print(f"Encoded sequence: {base_sequence_encoded}")

    base_dataset = Dataset_Prep(base_sequence_encoded, max_length)
    print("Padded/Truncated Sequences:", base_dataset.padded_sequences)

    # Split dataset into training and validation sets
    train_size = int(0.75 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create models
    generator = LSTMGenerator(noise_dim, hidden_dim, num_layers, max_length, vocab_size).to(device)
    discriminator = LSTMDiscriminator(hidden_dim, num_layers, max_length, vocab_size).to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0008)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001)

    # Train the models
    train(generator, discriminator, dataloader_train, dataloader_val, criterion, optimizer_g, optimizer_d, num_epochs, noise_dim, max_length, vocab_size, fig_name)

if __name__ == '__main__':
    main()
