#!/bin/bash
cd /home/stacys/src/masters_project/plasmids/5_generation_bakta/annotations

input_dir="/home/stacys/data/generated_sequences"
output_dir="/home/stacys/src/masters_project/plasmids/5_generation_bakta/annotations"

for fasta_file in "$input_dir"/*.fasta; do
    base_name=$(basename "$fasta_file" .fasta)

    # plannotate batch -i "$fasta_file" -o "$output_dir" --file_name "${base_name}"
    bakta --db /home/stacys/data/bakta_db --output "${base_name}" --prefix "${base_name}" --complete --verbose "$fasta_file"

    echo "Processed $fasta_file and saved output to $output_dirls"
done

echo "All files have been processed."
