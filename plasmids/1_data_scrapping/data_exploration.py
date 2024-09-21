## Gwenerating sequences size distribution
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

figures_dir = "/Users/anastasiiashcherbakova/git_projects/masters_project/figures/"

sequences = np.load('/Users/anastasiiashcherbakova/Desktop/1_data_scrapping/cleaned_sequences.npy', allow_pickle=True)
print(f"Sequences shape: {sequences.shape}")

sequence_lengths = np.array([len(seq) for seq in sequences])

mean = np.mean(sequence_lengths)
median = np.median(sequence_lengths)
min_value = np.min(sequence_lengths)
max_value = np.max(sequence_lengths)

plt.figure(figsize=(10, 8))
plt.hist(sequence_lengths, bins=20, color='darkorchid')
plt.xlabel('Plasmid size (bps)')
plt.ylabel('Frequency')
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
dummy_min = plt.Line2D([], [], color='black',  linewidth=2, label=f'Min: {min_value:.2f}')
dummy_max = plt.Line2D([], [], color='black', linewidth=2, label=f'Max: {max_value:.2f}')

handles = [plt.Line2D([], [], color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}'),
        plt.Line2D([], [], color='b', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}'),
        dummy_min, dummy_max]
plt.legend(handles=handles)
plt.savefig(figures_dir+"plasmid_sixe_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()



