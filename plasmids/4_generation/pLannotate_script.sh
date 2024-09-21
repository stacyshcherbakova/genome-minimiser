#!/bin/bash

input_dir="/home/stacys/data/generated_plasmid_sequences"
output_dir="/home/stacys/src/masters_project/plasmids/5_generation_plannotate/html_annotations"

for fasta_file in "$input_dir"/*.fasta; do
    base_name=$(basename "$fasta_file" .fasta)

    plannotate batch -i "$fasta_file" -o "$output_dir" --file_name "${base_name}" --html

    echo "Processed $fasta_file and saved output to $output_dirls"
done

echo "All files have been processed."
