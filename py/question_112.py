import numpy as np
import matplotlib.pyplot as plt
from question_111 import P_matrix as P

nucleotides = ['A', 'C', 'G', 'T']
# On observe l'évolution sur 20 étapes pour voir la convergence vers la stationnaire
t_max = 20

# Test sur deux conditions initiales différentes
u0_uniforme = np.array([0.25, 0.25, 0.25, 0.25])
u0_certain = np.array([0, 1, 0, 0])  # Départ certain sur 'C'


def calculer_evolution(u0, P, t_max):
    # Stocke l'évolution des probabilités (une ligne par étape temporelle)
    historique = np.zeros((t_max + 1, 4))
    historique[0] = u0

    for t in range(1, t_max + 1):
        # u_t = u_{t-1} * P  (ce qui équivaut à u_0 * P^t)
        historique[t] = np.dot(historique[t-1], P)
    return historique


hist_uniforme = calculer_evolution(u0_uniforme, P, t_max)
hist_certain = calculer_evolution(u0_certain, P, t_max)

# Création de deux graphiques pour comparer les deux trajectoires
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

couleurs = ['blue', 'orange', 'green', 'red']

# Plot 1: Départ Uniforme
for i in range(4):
    ax1.plot(
        range(t_max + 1),
        hist_uniforme[:, i],
        marker='o',
        label=nucleotides[i],
        color=couleurs[i]
    )
ax1.set_title("Évolution - Départ Uniforme")
ax1.set_xlabel("Temps (t)")
ax1.set_ylabel("Probabilité")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Départ 'C'
for i in range(4):
    ax2.plot(
        range(t_max + 1),
        hist_certain[:, i],
        marker='o',
        label=nucleotides[i],
        color=couleurs[i]
    )
ax2.set_title("Évolution - Départ 'C' (100%)")
ax2.set_xlabel("Temps (t)")
ax2.set_ylabel("Probabilité")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../images/evolution_markov.png')  # Sauvegarde l'image pour LaTeX
plt.show()

# Calcul de P^t pour un t grand (ex: t=50)
P_t50 = np.linalg.matrix_power(P, 50)
print("\nMatrice P^50 (t=50) :")
print(np.round(P_t50, 4))
