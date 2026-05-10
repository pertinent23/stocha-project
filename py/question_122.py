import numpy as np
import matplotlib.pyplot as plt

# Construction de la matrice cible (distribution jointe discrète 4x4)
# Formule : Q(i,j) = (i+j)/80 pour i,j dans {1,2,3,4}
# Cette distribution est strictement positive (garantit l'irréductibilité)
QY = np.fromfunction(lambda i, j: (i+1 + j+1)/80, (4, 4))


def sampler_inverse(probs):
    """Méthode de la transformée inverse"""
    U = np.random.uniform(0, 1)
    F = np.cumsum(probs)
    return np.searchsorted(F, U)


def get_cond_y1(y2, Q):
    # P(Y1 | Y2=y2)
    line = Q[:, y2]
    return line / np.sum(line)


def get_cond_y2(y1, Q):
    # P(Y2 | Y1=y1)
    line = Q[y1, :]
    return line / np.sum(line)


def simulation_gibbs(mode='random', n_iter=50000):
    counts = np.zeros((4, 4))
    y = [0, 0]  # État initial (indices 0 à 3)

    for _ in range(n_iter):
        if mode == 'random':
            j = np.random.choice([0, 1])
            if j == 0:
                y[0] = sampler_inverse(get_cond_y1(y[1], QY))
            else:
                y[1] = sampler_inverse(get_cond_y2(y[0], QY))
        else:  # Systematic
            y[0] = sampler_inverse(get_cond_y1(y[1], QY))
            y[1] = sampler_inverse(get_cond_y2(y[0], QY))

        counts[y[0], y[1]] += 1
    return counts / n_iter


# Simulations
freq_random = simulation_gibbs('random')
freq_syst = simulation_gibbs('systematic')

# Graphique de comparaison
labels = [f"({i+1},{j+1})" for i in range(4) for j in range(4)]
flat_Q = QY.flatten()
flat_R = freq_random.flatten()
flat_S = freq_syst.flatten()

x = np.arange(16)
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, flat_Q, width, label='Théorique (QY)', alpha=0.8)
ax.bar(x, flat_R, width, label='Aléatoire (Algo 1)', alpha=0.8)
ax.bar(x + width, flat_S, width, label='Systématique', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_title("Comparaison des fréquences : Gibbs 4x4")
ax.legend()
plt.tight_layout()
plt.savefig('../images/comparaison_gibbs.png')
plt.show()
