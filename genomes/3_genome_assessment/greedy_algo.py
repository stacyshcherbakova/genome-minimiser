from Bio import SeqIO
import numpy as np

genome_file = '/Users/anastasiiashcherbakova/git_projects/masters_project/data/sequence.gb'
essential_genes = np.load('/Users/anastasiiashcherbakova/git_projects/masters_project/data/essential_gene_in_ds.npy', allow_pickle=True)

gene_list = []
for record in SeqIO.parse(genome_file, 'genbank'):
    for feature in record.features:
        if feature.type == 'gene':
            gene_name = feature.qualifiers.get('gene', ['unknown'])[0]
            gene_length = len(feature.location)
            gene_list.append({'name': gene_name, 'length': gene_length})

gene_names = [gene['name'] for gene in gene_list]

gene_essential_mask = [gene in essential_genes for gene in gene_names]

non_essential_genes = [gene for gene, is_essential in zip(gene_list, gene_essential_mask) if not is_essential]

overlap = np.intersect1d(gene_names, essential_genes)
if overlap.size > 0:
    print(f"Overlap found: {overlap}")
    print(f"Total overlapping elements: {overlap.size}")
else:
    print("No overlap found")

reduction_log = []

def simplified_genome_reduction(non_essential_genes):
    reduced_genome = non_essential_genes.copy() 
    while non_essential_genes:
        gene_to_remove = select_gene_to_remove(non_essential_genes)
        reduced_genome.remove(gene_to_remove)
        non_essential_genes.remove(gene_to_remove)
        record_gene_removal(gene_to_remove)
    
    return reduced_genome

def select_gene_to_remove(non_essential_genes):
    return non_essential_genes[0]

def record_gene_removal(gene):
    print(f"Removed gene: {gene['name']} (Length: {gene['length']})")
    reduction_log.append(gene)

reduced_genome = simplified_genome_reduction(non_essential_genes)

print("Final reduced genome:")
for gene in reduced_genome:
    print(f"Gene: {gene['name']}, Length: {gene['length']}")
