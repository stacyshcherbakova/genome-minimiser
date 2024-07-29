### take the samples with the lowest number of genes and the highest number of essential genes 

### going theough the sample - if the genes of index i is present (equal to 1 at the eindex) we find the gene name from the columns (it will be th ith column) and then take the name and search the fasts file for the sequence)

### we find and add all the sequences that are present and create a new sequence

### OR

### we take the wild type of the e coli strain K12 MG1655, and go through the sample - if the genes of the index i is not present (equal to 0) we find the gene from the columns (it will be th ith column) and then take the name and search the fasts file for the sequnece

### we then take the sequence and delete it from the wild type 

### second approach is better 

from Bio import SeqIO
import numpy as np

sequence_file = "/Users/anastasiiashcherbakova/Downloads/sequence.gb" 
record = SeqIO.read(sequence_file, "genbank")
original_genome_length = len(record.seq)
# print(record)

needed_genes = np.load('/Users/anastasiiashcherbakova/git_projects/masters_project/data/unique_gene_names.npy', allow_pickle=True).tolist()
# print(features_to_remove)

non_essential_features = []
for feature in record.features:
    if feature.type == "gene":
        # print(f"Start: {feature.location.start}")
        # print(f"End: {feature.location.end}")
        gene_name = feature.qualifiers.get("gene", [""])[0]
        if gene_name not in needed_genes:
            non_essential_features.append(feature)

# print(non_essential_features)

positions_to_remove = set()
for feature in non_essential_features:
    positions_to_remove.update(range(int(feature.location.start), int(feature.location.end)))

# print(positions_to_remove)

new_sequence = ''.join(base for i, base in enumerate(record.seq) if i not in positions_to_remove)

with open("minimized_genome.fasta", "w") as output_file:
    output_file.write(">Minimized_E_coli_K12_MG1655\n")
    output_file.write(str(new_sequence))

print(f"Original genome length: {original_genome_length}")
print(f"Final minimized genome length: {len(new_sequence)}")