import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_models')))

from GANs_model_1 import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Basic GAN model using sngle nucleotide tokenisation 

def main():
    sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/cleaned_sequences.npy', allow_pickle=True) # ["ATCGTACG", "CGTACGTAGC", "ATCG", "CGTACGTAGCTAGCGT"] 
    max_length = 6000 # 10 
    batch_size = 4
    num_epochs = 1000 # 1
    noise_dim = 100

    base_tokenizer = Base_Level_Tokenizer()
    base_vocab = base_tokenizer.fit_on_texts(sequences)
    vocab_size = len(base_vocab)
    input_size = max_length * vocab_size
    print("Base Tokenizer Vocabulary:", base_vocab)

    base_sequence_encoded = base_tokenizer.sequence_to_indices(sequences)
    # print(f"Encoded sequence: {base_sequence_encoded}")

    base_dataset = Dataset_Prep(base_sequence_encoded, max_length)
    # print("Padded/Truncated Sequences:", base_dataset.padded_sequences)

    base_dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(noise_dim, max_length, vocab_size).to(device)
    discriminator = Discriminator(input_size).to(device)
    discriminator_conv = Conv_Discriminator(max_length, vocab_size).to(device)

    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)

    discriminator_losses = []
    discriminator_conv_losses = []
    generator_losses = []
    generator_losses_conv_disc = []

    for epoch in range(num_epochs):
        epoch_discriminator_loss = 0
        epoch_discriminator_conv_loss = 0
        epoch_generator_loss = 0
        epoch_generator_loss_with_conv_disc = 0
        count = 0

        for real_data in base_dataloader:
            real_data = real_data.to(device)
            batch_size_actual = real_data.size(0)
            count += 1

            real_data_one_hot = one_hot_encode_sequences(real_data, vocab_size).float()
            real_data_one_hot = real_data_one_hot.view(batch_size_actual, -1)

            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)

            optimizer_d.zero_grad()

            outputs_real = discriminator(real_data_one_hot)
            # print(real_data_one_hot)
            outputs_real_conv = discriminator_conv(real_data_one_hot)
            d_loss_real = criterion(outputs_real, real_labels)
            d_conv_loss_real = criterion(outputs_real_conv, real_labels)
            # print(f'Epoch {epoch+1}, Batch {i//batch_size + 1}:')
            # print(f'Discriminator output for real data: {outputs_real}')
            # print(f'Discriminator loss for real data: {d_loss_real.item()}')

            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise)
            fake_data = fake_data.view(batch_size_actual, -1)

            outputs_fake = discriminator(fake_data.detach())
            outputs_fake_conv = discriminator_conv(fake_data.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            d_conv_loss_fake = criterion(outputs_fake_conv, fake_labels)
            # print(f'Discriminator output for fake data: {outputs_fake}')
            # print(f'Discriminator loss for fake data: {d_loss_fake.item()}')

            d_loss = d_loss_real + d_loss_fake
            d_conv_loss = d_conv_loss_real + d_conv_loss_fake
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise)
            # fake_data_softmax_sum = fake_data.sum(dim=2)
            # print(f'Softmax sum for fake data (should be 1 for each element): {fake_data_softmax_sum}')
            # print(f"Generator output shape: {fake_data.shape}")

            fake_data = fake_data.view(batch_size_actual, -1)
            outputs = discriminator(fake_data)
            outputs_real_conv = discriminator_conv(fake_data)
            g_loss = criterion(outputs, real_labels) 
            g_loss_conv_disc = criterion(outputs_real_conv, real_labels) 

            # print(f'Discriminator output for fake data (Generator training): {outputs}')
            # print(f'Generator loss: {g_loss.item()}')

            g_loss.backward()
            optimizer_g.step()

            epoch_discriminator_loss += d_loss.item()
            epoch_discriminator_conv_loss += d_conv_loss.item()
            epoch_generator_loss += g_loss.item()
            epoch_generator_loss_with_conv_disc += g_loss_conv_disc.item()

        epoch_discriminator_loss /= count
        epoch_discriminator_conv_loss /= count
        epoch_generator_loss /= count
        epoch_generator_loss_with_conv_disc /= count 

        discriminator_losses.append(epoch_discriminator_loss)
        discriminator_conv_losses.append(epoch_discriminator_conv_loss)
        generator_losses.append(epoch_generator_loss)
        generator_losses_conv_disc.append(epoch_generator_loss_with_conv_disc)

        if epoch % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]\nDiscriminator loss: {epoch_discriminator_loss} \nConv Discriminator loss: {epoch_discriminator_conv_loss} \nGenerator loss: {epoch_generator_loss} \nGenerator loss with Conv Discriminator: {epoch_generator_loss_with_conv_disc}')
            print('-------------------------------------')

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(discriminator_conv_losses, label='Conv Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(generator_losses_conv_disc, label='Generator Loss with Conv Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig("/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/GAN_training_base_tok_losses.pdf", format="pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    torch.save(generator.state_dict(), '/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/saved_base_generator.pt') # /home/stacys/src/masters_project/plasmids
    torch.save(discriminator.state_dict(), '/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/saved_base_discriminator.pt') # /home/stacys/src/masters_project/plasmids
    torch.save(discriminator_conv.state_dict(), '/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/saved_base_conv_discriminator.pt') # /home/stacys/src/masters_project/plasmids

if __name__ == '__main__':
    main()

# base_dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

# kmer_tokenizer = KMerTokenizer(k)
# vocab = kmer_tokenizer.fit_on_texts(sequences)
# print("K-mer Tokenizer Vocabulary:", vocab)

# kmer_encoded_sequences = kmer_tokenizer.sequence_to_indices(sequences)
# print("Encoded Sequences:", kmer_encoded_sequences)

# kmer_dataset = Dataset_Prep(kmer_encoded_sequences, max_length)
# print("Padded/Truncated Sequences:", kmer_dataset.padded_sequences)

# # kmer_dataloader = DataLoader(kmer_dataset, batch_size=batch_size, shuffle=True)