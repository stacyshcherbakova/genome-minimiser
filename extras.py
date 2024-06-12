# Used libraries
import torch 
import numpy as np

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