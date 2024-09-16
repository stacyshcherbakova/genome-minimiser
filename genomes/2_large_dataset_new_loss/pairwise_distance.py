import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import to_tree

SAMPLES = 1000

presence_absence_matrix = np.load('/Users/anastasiiashcherbakova/git_projects/masters_project/data/additional_generated_samples.npy', allow_pickle=True).tolist()[:SAMPLES]

# print(presence_absence_matrix[:2])

print('1')
pairwise_dist = pdist(presence_absence_matrix, metric='jaccard')
print('2')
distance_matrix = squareform(pairwise_dist)
print('3')
linkage_matrix = linkage(pairwise_dist, method='average')
print('4')

plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, labels=[f'Sample {i+1}' for i in range(SAMPLES)])
plt.title('Genetic Distance Tree (UPGMA)')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

def to_newick(tree, labels):
    """ Recursively convert the scipy cluster tree into Newick format. """
    if tree.is_leaf():
        return labels[tree.id]
    else:
        return f"({to_newick(tree.get_left(), labels)},{to_newick(tree.get_right(), labels)})"

tree, nodes = to_tree(linkage_matrix, rd=True)
newick_str = to_newick(tree, [f'Sample {i+1}' for i in range(SAMPLES)]) + ";"

with open("upgma_tree.newick", "w") as f:
    f.write(newick_str)


