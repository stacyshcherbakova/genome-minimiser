from Bio import SeqIO
import os

input_fasta = "final_generated_sequences.fasta"
output_dir = "generated_sequences"

os.makedirs(output_dir, exist_ok=True)

for record in SeqIO.parse(input_fasta, "fasta"):
    output_file = os.path.join(output_dir, f"{record.id}.fasta")
    
    with open(output_file, "w") as output_handle:
        SeqIO.write(record, output_handle, "fasta")
    
    print(f"Saved {record.id} to {output_file}")
