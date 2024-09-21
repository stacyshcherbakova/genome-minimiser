import sys

model_dir = '/Users/anastasiiashcherbakova/git_projects/masters_project/plasmids/2_models'
sys.path.append(model_dir)

from GANs_model_1 import *
import torch
import torch
from torch.utils.data import DataLoader

max_length = 6000
vocab_size = 4
num_epochs = 100
noise_dim = 100
input_size = max_length * vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(noise_dim, max_length, vocab_size).to(device)
critic = Critic(input_size).to(device)  
# print(f"Type of critic before loading state dict: {type(critic)}")

generator.load_state_dict(torch.load('/Users/anastasiiashcherbakova/Desktop/GANs_model_1_output/saved_base_generator_mod.pt', map_location=torch.device('cpu')))
critic.load_state_dict(torch.load('/Users/anastasiiashcherbakova/Desktop/GANs_model_1_output/saved_base_critic_mod.pt', map_location=torch.device('cpu')))

generator.eval()
critic.eval()

first_linear_layer = generator.model[0]
latent_dim = first_linear_layer.weight.shape[1]
num_samples = 1000
random_latent_vectors = torch.randn(num_samples, latent_dim)

with torch.no_grad():
    generated_data = generator(random_latent_vectors)

generated_data = generated_data.cpu().numpy()
print(f"Generated data shape: {generated_data.shape}")

generated_data_tensor = torch.tensor(generated_data, dtype=torch.float32).to(device)

def tensor_to_sequences(tensor):
    nucleotide_map = {
        0: 'A',
        1: 'C',
        2: 'G',
        3: 'T',
    }
    
    max_indices = torch.argmax(tensor, dim=-1)  # dim=-1 selects the last dimension (4)
    
    tensor_np = max_indices.numpy()
    
    new_sequences = []
    for seq in tensor_np:
        sequence = ''.join([nucleotide_map[idx] for idx in seq])
        new_sequences.append(sequence)
    
    return new_sequences

new_sequences = tensor_to_sequences(generated_data_tensor)

with open('final_generated_sequences.fasta', 'w') as fasta_file:
    for i, seq in enumerate(new_sequences, 1):
        fasta_file.write(f">sequence_{i}\n")
        fasta_file.write(f"{seq}\n")

print(f"Number of generated sequences: {len(new_sequences)}")



