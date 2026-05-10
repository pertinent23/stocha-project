import numpy as np
from question_111 import P_matrix as P

# Calcul analytique de la distribution stationnaire
# Elle correspond au vecteur propre à gauche de P associé à la valeur propre 1

# On diagonalise P.T pour trouver les valeurs et vecteurs propres
valeurs_propres, vecteurs_propres_gauche = np.linalg.eig(P.T)

# Localisation de la valeur propre égale à 1 (attention aux erreurs numériques)
indice_vp_1 = np.where(np.isclose(valeurs_propres, 1.0))[0][0]

# Extraction du vecteur propre (eig retourne des nombres complexes, on prend la partie réelle)
vecteur_propre_1 = vecteurs_propres_gauche[:, indice_vp_1].real

# Normalisation pour obtenir une distribution de probabilité (somme = 1)
pi_infini = vecteur_propre_1 / np.sum(vecteur_propre_1)

print("Valeurs propres de P :", np.round(valeurs_propres.real, 4))
print("\nDistribution stationnaire analytique (pi_infini) :")
print(f"A: {pi_infini[0]:.4f}")
print(f"C: {pi_infini[1]:.4f}")
print(f"G: {pi_infini[2]:.4f}")
print(f"T: {pi_infini[3]:.4f}")
print("\nVecteur pi_infini :", np.round(pi_infini, 4))
