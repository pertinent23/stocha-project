import numpy as np
from root import seq1, seq2, seq3

from question_111 import P_matrix as P, nuc_to_idx
from question_113 import pi_infini


def calculer_frequences(sequence):
    comptages = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    for nuc in sequence:
        if nuc in comptages:
            comptages[nuc] += 1

    longueur = len(sequence)
    frequences = {nuc: count / longueur for nuc, count in comptages.items()}
    return frequences


sequences = {
    'Seq1': seq1.upper(),
    'Seq2': seq2.upper(),
    'Seq3': seq3.upper()
}

print("--- Fréquences d'apparition des nucléotides ---")
print(f"{'Séquence':<10} | {'A':<8} | {'C':<8} | {'G':<8} | {'T':<8}")
print("-" * 55)
for nom, seq in sequences.items():
    freqs = calculer_frequences(seq)
    print(
        f"{nom:<10} | {freqs['A']:.4f}   "
        f"| {freqs['C']:.4f}   | {freqs['G']:.4f}   | {freqs['T']:.4f}"
    )
print("\n")


def calculer_log_vraisemblance(sequence, matrice_P, dist_init):
    """
    Calcule la log-vraisemblance d'une séquence sous le modèle de Markov.
    On utilise le log pour éviter les débordements numériques avec les probabilités très petites.
    """
    # Gestion des probabilités à 0 : on ajoute un epsilon pour éviter log(0)
    log_P = np.log(np.maximum(matrice_P, 1e-10))
    log_pi = np.log(np.maximum(dist_init, 1e-10))

    # Log-probabilité du premier nucléotide (selon la stationnaire)
    premier_nuc = sequence[0]
    idx_premier = nuc_to_idx[premier_nuc]
    log_vraisemblance = log_pi[idx_premier]

    # Log-probabilités des transitions successives
    for i in range(len(sequence) - 1):
        nuc_courant = sequence[i]
        nuc_suivant = sequence[i+1]

        idx_courant = nuc_to_idx[nuc_courant]
        idx_suivant = nuc_to_idx[nuc_suivant]

        log_vraisemblance += log_P[idx_courant, idx_suivant]

    return log_vraisemblance


for seq_a_evaluer in [seq1.upper(), seq2.upper(), seq3.upper()]:
    log_l = calculer_log_vraisemblance(seq_a_evaluer, P, pi_infini)

    print(f"Longueur de la séquence : {len(seq_a_evaluer)}")
    print(f"Log-vraisemblance de la séquence : {log_l:.4f}")
    # Pour info, on peut calculer la log-vraisemblance moyenne par nucléotide
    print(
        f"Log-vraisemblance moyenne par nucléotide : "
        f"{log_l/len(seq_a_evaluer):.4f}"
    )
    print("\n")
