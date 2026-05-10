import numpy as np
from question_111 import P_matrix as P
from question_113 import pi_infini as pi_infini_theorique

nucleotides = ['A', 'C', 'G', 'T']

# Pour vérifier le théorème ergodique, on génère une longue trajectoire
# 50000 nucléotides suffisent pour avoir une bonne convergence statistique
N = 50000
etat_courant = 0  # Début arbitraire par 'A'
comptages = np.zeros(4)

# Simulation : on marche aléatoirement selon la chaîne de Markov
print(f"Génération d'une séquence de {N} nucléotides en cours...")
for _ in range(N):
    comptages[etat_courant] += 1
    # Le prochain nucléotide est tiré selon les probabilités de transition
    etat_courant = np.random.choice([0, 1, 2, 3], p=P[etat_courant])

# Calcul des fréquences empiriques d'apparition de chaque nucléotide
frequences_empiriques = comptages / N

# Affichage et comparaison avec la distribution stationnaire théorique
print("\nComparaison (Vérification du Théorème Ergodique) :")
header = (
    "Nucléotide | Fréq. Empirique | Prob. Stationnaire (pi) | "
    "Différence absolue"
)
print(header)
print("-" * len(header))

for i in range(4):
    nuc = nucleotides[i]
    freq = frequences_empiriques[i]
    pi_inf = pi_infini_theorique[i]
    diff = abs(freq - pi_inf)

    print(f"    {nuc}      |     {freq:.4f}      |       "
          f"{pi_inf:.4f}          |     {diff:.4f}")
