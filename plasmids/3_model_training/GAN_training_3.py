import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_models')))

from GANs_model_3 import *
from extras import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Using LSTM (long short term memroy) generato rna discriminator which tak einto account the previous data in the seqience - good for sequenctial data, was suggested by chat GPT

def train(generator, discriminator, dataloader_train, criterion, optimizer_g, optimizer_d, num_epochs, noise_dim, max_length, vocab_size, fig_name):
    train_discriminator_losses = []
    train_generator_losses = []
    train_real_accuracies = []
    train_fake_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_discriminator_loss = 0
        epoch_train_generator_loss = 0
        epoch_train_real_accuracy = 0
        epoch_train_fake_accuracy = 0
        train_count = 0

        for real_data in dataloader_train:
            real_data = real_data.to(device)
            batch_size_actual = real_data.size(0)

            real_data_one_hot = one_hot_encode_sequences(real_data, vocab_size)
            real_data_one_hot = real_data_one_hot.view(batch_size_actual, max_length, vocab_size)

            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise)
            fake_data = fake_data.view(batch_size_actual, max_length, vocab_size)

            d_loss, real_accuracy, fake_accuracy = train_discriminator(
                real_data_one_hot, fake_data, discriminator, optimizer_d, criterion)

            g_loss = train_generator(fake_data, discriminator, optimizer_g, criterion)

            epoch_train_discriminator_loss += d_loss
            epoch_train_generator_loss += g_loss
            epoch_train_real_accuracy += real_accuracy
            epoch_train_fake_accuracy += fake_accuracy
            train_count += 1

        epoch_train_discriminator_loss /= train_count
        epoch_train_generator_loss /= train_count
        epoch_train_real_accuracy /= train_count
        epoch_train_fake_accuracy /= train_count

        train_discriminator_losses.append(epoch_train_discriminator_loss)
        train_generator_losses.append(epoch_train_generator_loss)
        train_real_accuracies.append(epoch_train_real_accuracy)
        train_fake_accuracies.append(epoch_train_fake_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]\nTrain Discriminator loss: {epoch_train_discriminator_loss}\n'
              f'Train Generator loss: {epoch_train_generator_loss}\n'
              f'Train Real Accuracy: {epoch_train_real_accuracy*100:.2f}%\n'
              f'Train Fake Accuracy: {epoch_train_fake_accuracy*100:.2f}%')
        print('-------------------------')

    torch.save(generator.state_dict(), f'/home/stacys/src/masters_project/plasmids/2_models/GANs_model_2_output/saved_generator_{fig_name}.pt')
    torch.save(discriminator.state_dict(), f'/home/stacys/src/masters_project/plasmids/2_models/GANs_model_2_output/saved_discriminator_{fig_name}.pt')

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    plt.plot(train_discriminator_losses, label='Train Discriminator Loss')
    plt.plot(train_generator_losses, label='Train Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_real_accuracies, label='Train Real Accuracy')
    plt.plot(train_fake_accuracies, label='Train Fake Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("/home/stacys/src/masters_project/plasmids/2_models/GANs_model_2_output/GAN_training_" + fig_name + "_metrics.pdf", format="pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/cleaned_sequences.npy', allow_pickle=True)
    max_length = 6000 
    batch_size = 4
    num_epochs = 100 
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

    # Create dataloader
    dataloader_train = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

    # Create models
    generator = LSTMGenerator(noise_dim, hidden_dim, num_layers, max_length, vocab_size).to(device)
    discriminator = LSTMDiscriminator(hidden_dim, num_layers, max_length, vocab_size).to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0008)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001)

    # Train the models
    train(generator, discriminator, dataloader_train, criterion, optimizer_g, optimizer_d, num_epochs, noise_dim, max_length, vocab_size, fig_name)

if __name__ == '__main__':
    main()