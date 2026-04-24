import numpy as np
from question_111 import P_matrix as P
from question_113 import pi_infini as pi_infini_theorique

nucleotides = ['A', 'C', 'G', 'T']

# 2. Paramètres de la simulation
# Longueur de la séquence à générer
N = 50000
etat_courant = 0  # On commence arbitrairement par 'A' (indice 0)
comptages = np.zeros(4)

# 3. Génération de la séquence (Marche aléatoire)
print(f"Génération d'une séquence de {N} nucléotides en cours...")
for _ in range(N):
    comptages[etat_courant] += 1
    # On tire le prochain état en utilisant la ligne
    # de P correspondant à l'état courant
    etat_courant = np.random.choice([0, 1, 2, 3], p=P[etat_courant])

# 4. Calcul des fréquences empiriques
frequences_empiriques = comptages / N

# 5. Affichage et comparaison
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
