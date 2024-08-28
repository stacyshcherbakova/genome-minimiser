from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
plt.style.use('ggplot')

def pad_and_truncate_sequences(sequences, max_len):
    processed_sequences = []
    for seq in sequences:
        seq = list(seq)[:max_len]
        if len(seq) < max_len:
            seq += [''] * (max_len - len(seq))  
        processed_sequences.append(seq)
    result_array = np.array(processed_sequences, dtype='<U1')
    return result_array


### Loading the dataset
sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/cleaned_sequences.npy', allow_pickle=True)
NT_embeddings = np.load('/home/stacys/data/nucleotide_transformer_embeddings.npy', allow_pickle=True)


### Paddign and truncating sequences
### Does not work well
max_len = 2560
result_array = pad_and_truncate_sequences(sequences, max_len)
print(f"result_array shape: {result_array.shape}")
print(result_array)

### Encodign the sequences
char_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '': 4}
# split_sequences = [list(seq) for seq in sequences]

vectorized_map = np.vectorize(char_to_int.get)

int_sequences = [vectorized_map(seq) for seq in result_array]
# interpolated_seq = vectorized_map(result_array)

### Interpolate each sequence to the target length
# target_length = 2560

# interpolated_data = []
# for seq in int_sequences:
#     seq = list(seq)
#     x = np.linspace(0, 1, len(seq))
#     f = interp1d(x, seq, kind='linear') 
#     x_new = np.linspace(0, 1, target_length)
#     interpolated_seq = f(x_new)
#     interpolated_data.append(interpolated_seq)

# interpolated_data = np.array(interpolated_data)

### Standardize the data
scaler = StandardScaler()
interpolated_data_std = scaler.fit_transform(int_sequences)

### PCA on sequences
pca = PCA(n_components=2)
data_pca = pca.fit_transform(interpolated_data_std)
# columns = [f'PC{i+1}' for i in range(50)]
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1', y='PC2', data=df_pca)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
plt.savefig("pca_sequences.pdf", format="pdf", bbox_inches="tight")

### K-means clustering of PCAed sequences
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_pca)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(kmeans.cluster_centers_)

df_pca['Cluster'] = labels
plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=100, alpha=0.8)
# sns.scatterplot(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, label='Centers')
plt.savefig("kmeans_pca_sequences.pdf", format="pdf", bbox_inches="tight")

### PCA on embeddings
print(f"NT_embeddings shape: {NT_embeddings.shape}")

pca = PCA(n_components=2)
data_pca_embeddings = pca.fit_transform(NT_embeddings)
# columns = [f'PC{i+1}' for i in range(50)]
data_pca_embeddings = pd.DataFrame(data_pca_embeddings, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 10))
sns.scatterplot(x='PC1', y='PC2', data=data_pca_embeddings)
plt.savefig("pca_embeddings.pdf", format="pdf", bbox_inches="tight")