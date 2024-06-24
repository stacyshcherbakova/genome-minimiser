#!/bin/bash

## Extracting metadata for a small dataset 
# cd /Users/anastasiiashcherbakova/git_projects/masters_project/

# cut -d$'\t' -f1,9 /Users/anastasiiashcherbakova/Desktop/diss_track/Additional_File_3/Supplementary_File_1.txt > data/accessionID_phylogroup.txt


# head -n 1 accessionID_phylogroup.txt > data/accessionID_phylogroup2.txt

# tail -n +2 accessionID_phylogroup.txt | while IFS=$'\t' read -r AccessionID Phylogroup; do
#     AccessionID="${AccessionID:0:13}"
#     echo -e "$AccessionID\t$Phylogroup"
# done >> data/accessionID_phylogroup2.txt

# ## Extracting metadata for a big dataset 
# cd /Users/anastasiiashcherbakova/git_projects/masters_project/

# cut -d$',' -f1,12 /Users/anastasiiashcherbakova/Desktop/diss_track/F1_genome_metadata.csv > data/accessionID_phylogroup_BD.txt

## Extracting metadata for a big dataset FOR PCAs
cd /Users/anastasiiashcherbakova/git_projects/masters_project/

cut -d$',' -f1,4,6,7,9,11,12,17,19 /Users/anastasiiashcherbakova/Desktop/diss_track/F1_genome_metadata.csv > data/metadata_BD.txt


