import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_models')))

from GANs_model_4 import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient_penalty(netC, X_real_batch, X_fake_batch, device):
    batch_size = X_real_batch.shape[0]
    alpha = torch.rand(batch_size, 1, device=device).repeat(1, X_real_batch.shape[1])
    alpha = alpha.reshape(alpha.shape[0], 1, alpha.shape[1])
    interpolation = (alpha * X_real_batch + (1 - alpha) * X_fake_batch).requires_grad_(True)
    critic_interpolates = netC(interpolation)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolation,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


### Trying to replicate the GAN model from paper: Deep convolutional and conditional neural networks for large-scale genomic data generation - Yelmen et al 

def train_wgan_gp(generator, critic, dataloader, num_epochs, noise_dim, max_length, vocab_size):
    critic_losses = []
    generator_losses = []
    gps = []
    overlap_scores = []
    critic_iter=5
    lambda_gp=10

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))

    for epoch in range(num_epochs):
        for real_data in dataloader:
            batch_size_actual = real_data.size(0)
            real_data = real_data.to(device)
            real_data_one_hot = one_hot_encode_sequences(real_data, vocab_size)
            real_data_one_hot = real_data_one_hot.view(batch_size_actual, -1).float()
            # print(f'Shape of real_data: {real_data_one_hot.shape}') 
            
            for _ in range(critic_iter):
                noise = torch.randn(batch_size_actual, noise_dim).to(device).float()
                fake_data = generator(noise)

                # print(f'Shape of fake_data: {fake_data.shape}') 

                real_critic = critic(real_data_one_hot)
                fake_critic = critic(fake_data)

                
                gp = gradient_penalty(critic, real_data_one_hot, fake_data, device)
                
                critic_loss = -(torch.mean(real_critic) - torch.mean(fake_critic)) + lambda_gp * gp
                
                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()
            
            noise = torch.randn(batch_size_actual, noise_dim).to(device).float()
            fake_data = generator(noise)
            fake_critic = critic(fake_data.float())

            generator_loss = -torch.mean(fake_critic)
            
            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

            critic_losses.append(critic_loss.item())
            generator_losses.append(generator_loss.item())
            gps.append(gp.detach().item())

        print(f"Epoch [{epoch+1}/{num_epochs}]\nCritic Loss: {critic_loss.item()}\nGenerator Loss: {generator_loss.item()}")

        if epoch % 50 == 0:
             with torch.no_grad():
                noise = torch.randn(batch_size_actual, noise_dim).to(device).float()
                fake_data = generator(noise)
                real_data_sample = next(iter(dataloader)).to(device)
                real_data_one_hot = one_hot_encode_sequences(real_data_sample, vocab_size)
                real_data_one_hot = real_data_one_hot.view(batch_size_actual, -1).float()
                print(f'real_data_np shape: {real_data_one_hot.shape}') 
                print(f'fake_data_np shape: {fake_data.shape}')  
                real_pca, fake_pca = perform_pca(real_data_one_hot, fake_data)
                # plot_pca(real_pca, fake_pca)
                overlap_score = np.mean(np.abs(real_pca - fake_pca))
                overlap_scores.append(overlap_score)
                print(f'Overlap Score: {overlap_score}')

                if overlap_score < 0.1:
                    print(f'Early stopping at epoch {epoch+1}')
                    return
                
    plt.figure(figsize=(12, 16))

    plt.subplot(4, 1, 1)
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(overlap_scores, label='Overlap Score')
    plt.xlabel('Epoch / 50')
    plt.ylabel('Overlap Score')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(gps, label='Gradient Penalty')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Penalty')
    plt.legend()

    plt.savefig("GAN_training_4_metrics.pdf", format="pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def main():
    sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrappingcleaned_sequences.npy', allow_pickle=True)  # ["ATCGTACG", "CGTACGTAGC", "ATCG", "CGTACGTAGCTAGCGT"] # real sequences
    max_length = 6000
    noise_dim = 100
    num_epochs = 1000 #200 # we performed a second brief training (up to 200 epochs) with 10-fold lower generator learning rate (0.00005).
    batch_size = 2
    feature_size = max_length * vocab_size

    base_tokenizer = Base_Level_Tokenizer()
    base_vocab = base_tokenizer.fit_on_texts(sequences)
    vocab_size = len(base_vocab)
    # print("Base Tokenizer Vocabulary:", base_vocab)

    base_sequence_encoded = base_tokenizer.sequence_to_indices(sequences)
    # print(f"Encoded sequence: {base_sequence_encoded}")

    base_dataset = Dataset_Prep(base_sequence_encoded, max_length)
    # print("Padded/Truncated Sequences:", base_dataset.padded_sequences)

    dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(noise_dim, feature_size).to(device)
    critic = Critic(feature_size).to(device)

    train_wgan_gp(generator, critic, dataloader, num_epochs, noise_dim, max_length, vocab_size)

    torch.save(generator.state_dict(), '/Users/anastasiiashcherbakova/git_projects/masters_project/models/saved_wgan_generator.pt')
    torch.save(critic.state_dict(), '/Users/anastasiiashcherbakova/git_projects/masters_project/models/saved_wgan_critic.pt')

if __name__ == '__main__':
    main()
