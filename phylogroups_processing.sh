#!/bin/bash
cd /Users/anastasiiashcherbakova/git_projects/masters_project/

cut -d$'\t' -f1,9 /Users/anastasiiashcherbakova/Desktop/diss_track/Additional_File_3/Supplementary_File_1.txt > accessionID_phylogroup.txt


head -n 1 accessionID_phylogroup.txt > accessionID_phylogroup2.txt

tail -n +2 accessionID_phylogroup.txt | while IFS=$'\t' read -r AccessionID Phylogroup; do
    AccessionID="${AccessionID:0:13}"
    echo -e "$AccessionID\t$Phylogroup"
done >> accessionID_phylogroup2.txt


