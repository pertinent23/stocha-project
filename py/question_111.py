import numpy as np
from root import seq1

# Dictionnaire pour mapper les lettres aux indices (A=0, C=1, G=2, T=3)
seq1 = seq1.upper()  # Assurez-vous que la séquence est en majuscules
nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
P_counts = np.zeros((4, 4))

# Comptage des transitions
for i in range(len(seq1) - 1):
    current_nuc = seq1[i]
    next_nuc = seq1[i+1]
    P_counts[nuc_to_idx[current_nuc], nuc_to_idx[next_nuc]] += 1

# Division par la somme des lignes (MLE)
P_matrix = P_counts / P_counts.sum(axis=1, keepdims=True)

# Affichage pour LaTeX
print(np.round(P_matrix, 3))
