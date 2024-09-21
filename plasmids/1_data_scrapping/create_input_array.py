import os 
import numpy as np

# Modified the old downloaded plasmids doc by removing the names which i initially wanted to store
# downloaded_plasmids_file = os.path.join("/Users/anastasiiashcherbakova/Desktop/", 'downloaded_plasmids.txt')

# downloaded_plasmids = set()

# if os.path.exists(downloaded_plasmids_file):
#     with open(downloaded_plasmids_file, 'r') as f:
#         for line in f:
#             clean_line = line.split('\t')[0]
#             clean_line = clean_line.strip()
#             if clean_line:
#                 downloaded_plasmids.add(clean_line)

# print(len(downloaded_plasmids))

# with open('downloaded_plasmids.txt', 'w') as f:
#     for plasmid in downloaded_plasmids:
#         f.write(plasmid + '\n')



## Creating a single numpy array of plasmid sequences by removing the sequences with irrelecvant leters 
## and appendign them all into a npy array
directory = '/home/stacys/data/plasmid_sequences'

sequences = []
plasmid_ids = []

unwanted_letters = set('-DKMNRSVWY')

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        plasmid_id = filename.split('.')[0]

        if plasmid_id != 'downloaded_plasmids' and plasmid_id != 'last_processed_page':
            with open(os.path.join(directory, filename), 'r') as file:
                sequence = ''

                for line in file:
                    if line.startswith('>'):
                        continue
                    sequence += line.strip().upper()

                if not any(letter in sequence for letter in unwanted_letters):
                    sequences.append(sequence)
                    plasmid_ids.append(plasmid_id)

sequences_array = np.array(sequences, dtype=object)
plasmid_ids_array = np.array(plasmid_ids).astype(int)

np.save('cleaned_sequences.npy', sequences_array)
np.save('cleaned_plasmid_ids.npy', plasmid_ids_array)



## Check the ditribution of sequence lengths
# sequences = np.load('/home/stacys/src/masters_project/plasmids/1_data_scrapping/all_sequences.npy', allow_pickle=True)

# seq_lengths = []

# for sequence in sequences:
#     seq_leg = len(sequence)
#     seq_lengths.append(seq_leg)