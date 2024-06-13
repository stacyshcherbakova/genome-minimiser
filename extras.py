# Used libraries
import pandas as pd 
import torch 
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract latent variables
def get_latent_variables(model, data_loader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(data_loader):
            data = data.to(device)
            mean, logvar = model.encode(data)
            latents.append(mean.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    return latents

def do_tsne(n_components, latents, fig_name):
    tsne = TSNE(n_components=n_components)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], color='dodgerblue')
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