#!/bin/bash

input_dir="/home/stacys/data/generated_plasmid_sequences"
output_dir="/home/stacys/src/masters_project/plasmids/5_generation_plannotate/html_annotations"

plasmids=(
    'sequence_614'
    'sequence_58'
    'sequence_240'
    'sequence_507'
    'sequence_24'
    'sequence_587'
    'sequence_599'
    'sequence_571'
    'sequence_279'
    'sequence_729'
    'sequence_345'
    'sequence_475'
    'sequence_110'
    'sequence_362'
    'sequence_566'
    'sequence_602'
    'sequence_221'
    'sequence_768'
    'sequence_845'
    'sequence_972'
    'sequence_574'
    'sequence_187'
    'sequence_250'
    'sequence_786'
    'sequence_509'
    'sequence_868'
    'sequence_400'
    'sequence_528'
    'sequence_629'
    'sequence_516'
)

for plasmid in "${plasmids[@]}"; do
    fasta_file="${input_dir}/${plasmid}.fasta"
    base_name=$(basename "$fasta_file" .fasta)

    plannotate batch -i "$fasta_file" -o "$output_dir" --file_name "${base_name}" --html

    echo "Processed $fasta_file and saved HTML output to $output_dir"
done

echo "All specified files have been processed."

# for fasta_file in "$input_dir"/*.fasta; do
#     base_name=$(basename "$fasta_file" .fasta)

#     plannotate batch -i "$fasta_file" -o "$output_dir" --file_name "${base_name}"

#     echo "Processed $fasta_file and saved output to $output_dirls"
# done

# echo "All files have been processed."
