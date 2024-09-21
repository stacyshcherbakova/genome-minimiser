from Bio import SeqIO
from collections import defaultdict

# Read sequences from a FASTA file
file_path = "/Users/anastasiiashcherbakova/git_projects/masters_project/data/generated_genomes.fasta"  # Replace with your file
sequences = list(SeqIO.parse(file_path, "fasta"))

# Create a dictionary to store sequence and corresponding IDs
sequence_dict = defaultdict(list)

# Populate the dictionary: sequence as the key, list of record IDs as the value
for record in sequences:
    sequence_dict[str(record.seq)].append(record.id)

# Check for identical sequences
duplicates = {seq: ids for seq, ids in sequence_dict.items() if len(ids) > 1}

# Output duplicates
if duplicates:
    print("Identical sequences found:")
    for seq, ids in duplicates.items():
        print(f"Sequence: {seq[:30]}... (truncated) is shared by records: {', '.join(ids)}")
else:
    print("No identical sequences found.")
