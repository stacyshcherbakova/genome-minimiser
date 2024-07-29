# Used libraries
import pandas as pd 
import torch 
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from VAE_models.VAE_model import *
from VAE_models.VAE_model_enhanced import *

def count_essential_genes(binary_generated_samples, essential_gene_positions):

    binary_generated_samples = binary_generated_samples.astype(int)

    essential_genes_count_per_sample = np.zeros(10000, dtype=int)

    for sample_index in range(10000):
        present_essential_genes = 0
        
        for gene, positions in essential_gene_positions.items():
            if len(positions) == 1:
                pos = positions[0]
                if pos < binary_generated_samples.shape[1]:
                    if binary_generated_samples[sample_index, pos] != 0:
                        present_essential_genes += 1
            else:
                for pos in positions:
                    if pos < binary_generated_samples.shape[1]:
                        if binary_generated_samples[sample_index, pos] != 0:
                            present_essential_genes += 1
                            break

        essential_genes_count_per_sample[sample_index] = present_essential_genes

    return essential_genes_count_per_sample

def plot_essential_genes_distribution(binary_generated_samples, figure_name, plot_color):
    mean = np.mean(binary_generated_samples)
    median = np.median(binary_generated_samples)
    min_value = np.min(binary_generated_samples)
    max_value = np.max(binary_generated_samples)

    plt.figure(figsize=(10,10))
    plt.hist(binary_generated_samples, bins=10, color=plot_color)
    plt.xlabel('Genome size')
    plt.ylabel('Frequency')

    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

    handles = [plt.Line2D([], [], color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}'),
            plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'),
            dummy_min, dummy_max]

    plt.legend(handles=handles)

    plt.savefig(figure_name, format="pdf", bbox_inches="tight")

def plot_samples_distribution(binary_generated_samples, figure_name, plot_color):
    samples_size_sum = binary_generated_samples.sum(axis=1)

    mean = np.mean(samples_size_sum)
    median = np.median(samples_size_sum)
    min_value = np.min(samples_size_sum)
    max_value = np.max(samples_size_sum)

    plt.figure(figsize=(10,10))
    plt.hist(samples_size_sum, bins=10, color=plot_color)
    plt.xlabel('Genome size')
    plt.ylabel('Frequency')

    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
    dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

    handles = [plt.Line2D([], [], color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}'),
            plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'),
            dummy_min, dummy_max]

    plt.legend(handles=handles)

    plt.savefig(figure_name, format="pdf", bbox_inches="tight")

def load_model_enhanced(input_dim, hidden_dim, latent_dim, path_to_model):
    # Load trained model 
    input_dim = input_dim
    hidden_dim = hidden_dim
    latent_dim = latent_dim

    # changes layer norm layer to batch norm layer and 
    model = VAE_enhanced(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))  
    model.eval()  

    # Generate 10 new samples
    num_samples = 10000
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)  # Sample from the standard normal distribution because the latent space follows normal distribution 
        generated_samples = model.decode(z).cpu().numpy() 

    threshold = 0.5
    binary_generated_samples = (generated_samples > threshold).astype(float)

    print("Generated samples (binary):\n", binary_generated_samples)
    print("\n")
    print("Generated samples (sigmoid function output):\n", generated_samples)

    return model, binary_generated_samples


def load_model(input_dim, hidden_dim, latent_dim, path_to_model):
    # Load trained model 
    input_dim = input_dim
    hidden_dim = hidden_dim
    latent_dim = latent_dim

    # changes layer norm layer to batch norm layer and 
    model = VAE(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))  
    model.eval()  

    # Generate 10 new samples
    num_samples = 10000
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)  # Sample from the standard normal distribution because the latent space follows normal distribution 
        generated_samples = model.decode(z).cpu().numpy() 

    threshold = 0.5
    binary_generated_samples = (generated_samples > threshold).astype(float)

    print("Generated samples (binary):\n", binary_generated_samples)
    print("\n")
    print("Generated samples (sigmoid function output):\n", generated_samples)

    return model, binary_generated_samples

def l1_regularization(model, lambda_l1):
    l1_penalty = 0.0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty

def cosine_annealing_schedule(t, T, min_beta, max_beta):
    return min_beta + (max_beta - min_beta) / 2 * (1 + np.cos(np.pi * (t % T) / T))

def exponential_decay_schedule(t, initial_beta, decay_rate):
    return initial_beta * np.exp(-decay_rate * t)

# Function to extract latent variables
def get_latent_variables(model, data_loader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(torch.float).to(device)
            mean, logvar = model.encode(data)
            latents.append(mean.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents

def do_tsne(n_components, latents, fig_name):
    tsne = TSNE(n_components=n_components)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], color='dodgerblue')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()

def do_pca(n_components, latents, fig_name):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(latents)
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='PC1', y='PC2', data=df_pca, color='dodgerblue')
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()

def plot_loss_vs_epochs_graph(epochs, train_loss_vals, val_loss_vals, fig_name):
    plt.figure(figsize=(10,8))
    plt.scatter(epochs, train_loss_vals, color='dodgerblue')
    plt.plot(epochs, train_loss_vals, label='Train Loss', color='dodgerblue')
    plt.scatter(epochs, val_loss_vals, color='darkorange')
    plt.plot(epochs, val_loss_vals, label='Validation Loss', color='darkorange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()
