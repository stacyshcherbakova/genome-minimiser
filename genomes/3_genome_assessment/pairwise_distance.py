import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
import matplotlib.pyplot as plt

SAMPLES = 1000

large_data = pd.read_csv("/Users/anastasiiashcherbakova/git_projects/masters_project/data/F4_complete_presence_absence.csv", index_col=[0], header=[0])
phylogroup_data = pd.read_csv("/Users/anastasiiashcherbakova/git_projects/masters_project/data/accessionID_phylogroup_BD.csv", index_col=[0], header=[0])
data_without_lineage = large_data.drop(index=['Lineage'])
large_data_t = np.array(data_without_lineage.transpose())
merged_df = pd.merge(data_without_lineage.transpose(), phylogroup_data, how='inner', left_index=True, right_on='ID')
data_array_t = np.array(merged_df.iloc[:, :-1]).tolist()[:SAMPLES]
phylogroups_array = np.array(merged_df.iloc[:, -1])


# presence_absence_matrix = np.load('/Users/anastasiiashcherbakova/git_projects/masters_project/data/additional_generated_samples.npy', allow_pickle=True).tolist()[:SAMPLES]

pairwise_dist = pdist(data_array_t, metric='jaccard')

distance_matrix = squareform(pairwise_dist)

linkage_matrix = linkage(pairwise_dist, method='average')

def to_newick(tree, labels, parent_dist=0):
    """ Recursively convert the scipy cluster tree into Newick format with edge lengths. """
    if tree.is_leaf():
        return f"{labels[tree.id]}:{parent_dist - tree.dist}"
    else:
        left = to_newick(tree.get_left(), labels, tree.dist)
        right = to_newick(tree.get_right(), labels, tree.dist)
        return f"({left},{right}):{parent_dist - tree.dist}"

tree, nodes = to_tree(linkage_matrix, rd=True)
newick_str = to_newick(tree, [f'Sample {i+1}' for i in range(SAMPLES)]) + ";"

with open("/Users/anastasiiashcherbakova/git_projects/masters_project/data/upgma_tree_final_dataset.newick", "w") as f:
    f.write(newick_str)

np.savetxt('/Users/anastasiiashcherbakova/git_projects/masters_project/data/pairwise_distances_final_dataset.csv', distance_matrix, delimiter=',')
