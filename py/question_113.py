import numpy as np
from question_111 import P_matrix as P

# 1. Calcul des valeurs propres et vecteurs propres de la matrice transposée
valeurs_propres, vecteurs_propres_gauche = np.linalg.eig(P.T)

# 2. Trouver l'indice de la valeur propre égale à 1
# (On utilise np.isclose pour éviter les erreurs d'arrondi des flottants)
indice_vp_1 = np.where(np.isclose(valeurs_propres, 1.0))[0][0]

# 3. Extraire le vecteur propre correspondant
# (eig peut renvoyer des nombres complexes
# avec une partie imaginaire nulle, on prend .real)
vecteur_propre_1 = vecteurs_propres_gauche[:, indice_vp_1].real

# 4. Normaliser le vecteur pour que la somme de ses
# composantes vaille 1 (loi de probabilité)
pi_infini = vecteur_propre_1 / np.sum(vecteur_propre_1)

print("Valeurs propres de P :", np.round(valeurs_propres.real, 4))
print("\nDistribution stationnaire analytique (pi_infini) :")
print(f"A: {pi_infini[0]:.4f}")
print(f"C: {pi_infini[1]:.4f}")
print(f"G: {pi_infini[2]:.4f}")
print(f"T: {pi_infini[3]:.4f}")
print("\nVecteur pi_infini :", np.round(pi_infini, 4))
