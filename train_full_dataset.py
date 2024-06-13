# Import all the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from VAE_model import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from training import *
from extras import *
plt.style.use('ggplot')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("** START OF THE SCRIPT **\n")

## Loading and preping the dataset
print("LOADING THE DATASET...")
large_data = pd.read_csv("F4_complete_presence_absence.csv", index_col=[0], header=[0])

large_data_t = np.array(large_data.transpose())
large_data_t = large_data_t[:,1:]
print(f"Dataset shape: {large_data_t.shape}")

## Preping the dataset
print("PREPING THE DATASET...")
# Convert to PyTorch tensor
data_tensor = torch.tensor(large_data_t, dtype=torch.float32)

# Split into train and test sets
train_data, val_data = train_test_split(data_tensor, test_size=0.2, random_state=12345)
train_data, test_data = train_test_split(data_tensor, test_size=0.25, random_state=12345)

# TensorDataset
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)

# DataLoaders for main training
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

## Preping model inputs
print("PREPPING MODEL INPUTS...")
# Model inputs 
hidden_dim = 512
latent_dim = 64
beta_start = 0.1
beta_end = 1.0
n_epochs = 10
max_norm = 1.0 
input_dim = large_data_t.shape[1]

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

## Trainign the model
print("TRAINING STARTED...")
train_loss_vals2, val_loss_vals = train_with_KL_annelaing(model=model, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs, train_loader=train_loader, val_loader=val_loader, beta_start=beta_start, beta_end=beta_end, max_norm=max_norm)

# Save trained model
torch.save(model.state_dict(), "saved_KL_annealing_VAE_BD.pt")
print("Model saved.")

## Generating a comparison graph 
print("GENERATING A COMPARISON GRAPH...")
# Generating points for graphs
epochs = np.linspace(1, n_epochs, num=n_epochs)
# Plot train vs val loss graph
name = "second_model_train_val_loss_BD.pdf"
plot_loss_vs_epochs_graph(epochs=epochs, train_loss_vals=train_loss_vals2, val_loss_vals=val_loss_vals, fig_name=name)


## Exploring latent space
print("EXPLORING THE LATENT SPACE...")
# Get latent variables
latents = get_latent_variables(model, train_loader, device)

# Apply t-SNE for dimensionality reduction
name = "tsne_latent_space_visualisation_BD.pdf"
do_tsne(n_components=2, latents=latents, fig_name=name)

# Apply PCA
name = "pca_latent_space_visualisation_BD.pdf"
do_pca(n_components=2, latents=latents, fig_name=name)

print("\n** SCRIPT IS DONE **")