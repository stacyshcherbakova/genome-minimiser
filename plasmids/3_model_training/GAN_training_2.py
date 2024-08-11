import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_models')))

from GANs_model_2 import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Trying to replicate the GN model from paper: Creating artificial human genomes using generative neural networks - Yelmen et al

def train(generator, discriminator, dataloader_train, dataloader_val, criterion, optimizer_g, optimizer_d, num_epochs, noise_dim, max_length, vocab_size, fig_name):
    train_discriminator_losses = []
    train_generator_losses = []
    val_discriminator_losses = []
    val_generator_losses = []
    
    for epoch in range(num_epochs):
        epoch_train_discriminator_loss = 0
        epoch_train_generator_loss = 0
        epoch_val_discriminator_loss_real = 0
        epoch_val_discriminator_loss_fake = 0
        epoch_val_generator_loss = 0
        train_count = 0
        val_count = 0
        
        for real_data in dataloader_train:
            print('log')
            real_data = real_data.to(device)
            batch_size_actual = real_data.size(0)

            real_data_one_hot = one_hot_encode_sequences(real_data, vocab_size)
            real_data_one_hot = real_data_one_hot.view(batch_size_actual, max_length, vocab_size)

            real_labels = smooth_labels(torch.ones(batch_size_actual, 1).to(device))
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)

            optimizer_d.zero_grad()

            outputs_real = discriminator(real_data_one_hot)
            d_loss_real = criterion(outputs_real, real_labels)

            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise)
            fake_data = fake_data.view(batch_size_actual, max_length, vocab_size)
            outputs_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise)
            fake_data = fake_data.view(batch_size_actual, max_length, vocab_size)
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, torch.ones(batch_size_actual, 1).to(device))  

            g_loss.backward()
            optimizer_g.step()

            epoch_train_discriminator_loss += d_loss.item()
            epoch_train_generator_loss += g_loss.item()
            train_count += 1

        if epoch % 100 == 0:
            with torch.no_grad():
                for val_data in dataloader_val:
                    val_data = val_data.to(device)
                    batch_size_val = val_data.size(0)

                    val_data_one_hot = one_hot_encode_sequences(val_data, vocab_size)
                    val_data_one_hot = val_data_one_hot.view(batch_size_val, max_length, vocab_size)

                    real_labels_val = torch.ones(batch_size_val, 1).to(device)
                    fake_labels_val = torch.zeros(batch_size_val, 1).to(device)

                    outputs_real_val = discriminator(val_data_one_hot)
                    d_val_loss_real = criterion(outputs_real_val, real_labels_val)

                    noise_val = torch.randn(batch_size_val, noise_dim).to(device)
                    fake_data_val = generator(noise_val)
                    # print("Generated data shape:", fake_data_val.shape)
                    fake_data_val = fake_data_val.view(batch_size_val, max_length, vocab_size)
                    outputs_fake_val = discriminator(fake_data_val)
                    d_val_loss_fake = criterion(outputs_fake_val, fake_labels_val)

                    outputs_val = discriminator(fake_data_val)
                    g_val_loss = criterion(outputs_val, real_labels_val)

                    epoch_val_discriminator_loss_real += d_val_loss_real.item()
                    epoch_val_discriminator_loss_fake += d_val_loss_fake.item()
                    epoch_val_generator_loss += g_val_loss.item()
                    val_count += 1

                epoch_val_discriminator_loss_real /= val_count
                epoch_val_discriminator_loss_fake /= val_count
                epoch_val_generator_loss /= val_count

        epoch_train_discriminator_loss /= train_count
        epoch_train_generator_loss /= train_count

        train_discriminator_losses.append(epoch_train_discriminator_loss)
        train_generator_losses.append(epoch_train_generator_loss)
        val_discriminator_losses.append(epoch_val_discriminator_loss_real + epoch_val_discriminator_loss_fake)
        val_generator_losses.append(epoch_val_generator_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}]\nTrain Discriminator loss: {epoch_train_discriminator_loss}\nTrain Generator loss: {epoch_train_generator_loss}\nVal Discriminator real loss: {epoch_val_discriminator_loss_real}\nVal Discriminator fake loss: {epoch_val_discriminator_loss_fake}\nVal Generator loss: {epoch_val_generator_loss}')
        print('-------------------------')

    torch.save(generator.state_dict(), f'/home/stacys/src/masters_project/plasmids/2_models/GANs_model_2_output/saved_generator_{fig_name}.pt') # /home/stacys/src/masters_project/plasmids
    torch.save(discriminator.state_dict(), f'/home/stacys/src/masters_project/plasmids/2_models/GANs_model_2_output/saved_discriminator_{fig_name}.pt') # /home/stacys/src/masters_project/plasmids

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_discriminator_losses, label='Train Discriminator Loss')
    plt.plot(val_discriminator_losses, label='Val Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_generator_losses, label='Train Generator Loss')
    plt.plot(val_generator_losses, label='Val Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig("/home/stacys/src/masters_project/plasmids/2_models/GANs_model_2_output/GAN_training_" + fig_name + "_losses.pdf", format="pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def main():
    sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/cleaned_sequences.npy', allow_pickle=True) # ["ATCGTACG", "CGTACGTAGC", "ATCG", "CGTACGTAGCTAGCGT"] # real sequences
    max_length = 6000
    batch_size = 2
    num_epochs = 1000
    noise_dim = 600
    fig_name = "yelmen_1"

    base_tokenizer = Base_Level_Tokenizer()
    base_vocab = base_tokenizer.fit_on_texts(sequences)
    vocab_size = len(base_vocab)
    print("Base Tokenizer Vocabulary:", vocab_size)

    base_sequence_encoded = base_tokenizer.sequence_to_indices(sequences)
    # print(f"Encoded sequence: {base_sequence_encoded}")

    base_dataset = Dataset_Prep(base_sequence_encoded, max_length)
    # print("Padded/Truncated Sequences:", base_dataset.padded_sequences)

    train_size = int(0.75 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    generator = Generator(noise_dim, max_length, vocab_size).to(device)
    discriminator = Discriminator(max_length, vocab_size).to(device)

    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0008)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, weight_decay=0.0001)

    print("Trainign started")
    train(generator, discriminator, dataloader_train, dataloader_val, criterion, optimizer_g, optimizer_d, num_epochs, noise_dim, max_length, vocab_size, fig_name)

if __name__ == '__main__':
    main()