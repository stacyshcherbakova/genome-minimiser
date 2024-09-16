import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_models')))

from GANs_model_1 import *
from genomes.extras import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Basic GAN model using single nucleotide tokenization

def main():
    sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/cleaned_sequences.npy', allow_pickle=True)
    extra_name = 'base'
    max_length = 6000
    batch_size = 4
    num_epochs = 100
    noise_dim = 100

    base_tokenizer = Base_Level_Tokenizer()
    base_vocab = base_tokenizer.fit_on_texts(sequences)
    vocab_size = len(base_vocab)
    input_size = max_length * vocab_size
    print("Base Tokenizer Vocabulary:", base_vocab)

    base_sequence_encoded = base_tokenizer.sequence_to_indices(sequences)
    base_dataset = Dataset_Prep(base_sequence_encoded, max_length)
    base_dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(noise_dim, max_length, vocab_size).to(device)
    discriminator = Discriminator(input_size).to(device)

    criterion = nn.BCELoss()
    lr_g = 0.0002 * 4 
    lr_d = 0.0001 
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g)

    discriminator_losses = []
    generator_losses = []
    real_data_accuracies = []
    fake_data_accuracies = []

    for epoch in range(num_epochs):
        epoch_discriminator_loss = 0
        epoch_generator_loss = 0
        epoch_real_accuracy = 0
        epoch_fake_accuracy = 0
        count = 0

        for real_data in base_dataloader:
            real_data = real_data.to(device)
            batch_size_actual = real_data.size(0)
            count += 1

            real_data_one_hot = one_hot_encode_sequences(real_data, vocab_size).float()
            real_data_one_hot = real_data_one_hot.view(batch_size_actual, -1)

            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise)
            fake_data = fake_data.view(batch_size_actual, -1)

            d_loss, real_accuracy, fake_accuracy = train_discriminator(real_data_one_hot, fake_data, discriminator, optimizer_d, criterion)
            g_loss = train_generator(fake_data, discriminator, optimizer_g, criterion)

            epoch_discriminator_loss += d_loss
            epoch_generator_loss += g_loss
            epoch_real_accuracy += real_accuracy
            epoch_fake_accuracy += fake_accuracy

        epoch_discriminator_loss /= count
        epoch_generator_loss /= count
        epoch_real_accuracy /= count
        epoch_fake_accuracy /= count

        discriminator_losses.append(epoch_discriminator_loss)
        generator_losses.append(epoch_generator_loss)
        real_data_accuracies.append(epoch_real_accuracy)
        fake_data_accuracies.append(epoch_fake_accuracy)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]\nDiscriminator loss: {epoch_discriminator_loss} \nGenerator loss: {epoch_generator_loss} \nReal data accuracy: {epoch_real_accuracy*100:.2f}% \nFake data accuracy: {epoch_fake_accuracy*100:.2f}%')
            print('-------------------------------------')

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(real_data_accuracies, label='Real Data Accuracy')
    plt.plot(fake_data_accuracies, label='Fake Data Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/GAN_training_base_tok_metrics_base.pdf", format="pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    torch.save(generator.state_dict(), '/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/saved_base_generator.pt')
    torch.save(discriminator.state_dict(), '/home/stacys/src/masters_project/plasmids/2_models/GANs_model_1_output/saved_base_discriminator.pt')

if __name__ == '__main__':
    main()
