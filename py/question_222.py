import numpy as np


def load_sequences(filename):
    """
    Charge les séquences à partir d'un fichier texte (format type FASTA).

    Args:
        filename (str): Chemin vers le fichier contenant les séquences.

    Returns:
        list: Une liste de chaînes de caractères (séquences en majuscules).
    """
    sequences = []
    with open(filename, 'r') as f:
        current_seq = ""
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq.upper())
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq.upper())
    return sequences


def get_counts(sequences, positions, W):
    """
    Calcule la matrice des occurrences des nucléotides pour chaque colonne.
    Cette fonction représente la Vraisemblance (comptage des données).

    Args:
        sequences (list): Liste des séquences d'ADN.
        positions (list): Liste des positions actuelles des motifs.
        W (int): Longueur du motif.

    Returns:
        np.ndarray: Matrice de taille (4, W) contenant les comptages n_ij.
    """
    counts = np.zeros((4, W))
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, pos in enumerate(positions):
        motif_seq = sequences[i][pos:pos + W]
        for j, char in enumerate(motif_seq):
            counts[nuc_to_idx[char], j] += 1
    return counts


def sample_theta(counts, alpha=1.0):
    """
    Échantillonne le profil du motif selon la distribution a posteriori.
    Utilise la conjugaison Dirichlet-Multinomiale pour obtenir les paramètres mis à jour.
    """
    W = counts.shape[1]
    theta = np.zeros((4, W))
    for j in range(W):
        # Chaque colonne est échantillonnée indépendamment
        # Dirichlet(alpha + comptages) est la distribution a posteriori
        # Le pseudo-compte alpha=1 correspond au prior de Laplace (lissage uniforme)
        theta[:, j] = np.random.dirichlet(counts[:, j] + alpha)
    return theta


def sample_positions(sequences, theta, phi, W):
    """
    Échantillonne les positions du motif dans chaque séquence.
    Calcule le ratio vraisemblance (motif / fond) pour chaque position possible.
    """
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    new_positions = []

    for seq in sequences:
        L = len(seq)
        possible_starts = L - W + 1
        weights = np.zeros(possible_starts)

        for a in range(possible_starts):
            # Ratio Motif/Fond : produit des ratios pour chaque position
            w = 1.0
            for j in range(W):
                nuc = seq[a + j]
                idx = nuc_to_idx[nuc]
                w *= (theta[idx, j] / phi[idx])
            weights[a] = w

        # Normalisation des poids en probabilités
        prob = weights / np.sum(weights)

        # Tirage selon la distribution (transformée inverse implicite avec np.random.choice)
        choix = np.random.choice(possible_starts, p=prob)
        new_positions.append(choix)

    return new_positions


def phase_shift_move(sequences, positions, theta, phi, W):
    """
    Tente un décalage global des positions pour échapper aux minima locaux.
    C'est un mouvement Metropolis-Hastings avec probabilité d'acceptation 30%.
    """
    # On propose un décalage de +1 ou -1 pour tous les motifs
    shift = np.random.choice([-1, 1])
    new_pos = [p + shift for p in positions]

    # Vérification que le décalage reste dans les limites des séquences
    for i, p in enumerate(new_pos):
        if p < 0 or p + W > len(sequences[i]):
            return positions  # Rejet du mouvement

    # Acceptation avec une probabilité de 30% (heuristique empirique)
    if np.random.rand() < 0.3:
        return new_pos
    return positions


def gibbs_motif_discovery(seq_file, W, phi, iterations=1000):
    """
    Implémentation de l'échantillonneur de Gibbs pour la découverte de motifs.
    Alterne entre l'échantillonnage du profil et celui des positions.
    """
    sequences = load_sequences(seq_file)

    # Initialisation aléatoire des positions
    positions = [np.random.randint(0, len(s) - W + 1) for s in sequences]

    for i in range(iterations):
        # Étape A : Échantillonner le profil de motif
        counts = get_counts(sequences, positions, W)
        theta = sample_theta(counts)

        # Étape B : Échantillonner les nouvelles positions
        positions = sample_positions(sequences, theta, phi, W)

        # Mouvement de phase shift tous les 10 itérations
        # Permet d'éviter les alignements décalés (optima locaux)
        if i % 10 == 0:
            positions = phase_shift_move(
                sequences, positions, theta, phi, W
            )

        # Affichage de la progression
        if i % 100 == 0:
            print(f"Itération {i}...")

    return positions, theta


# Paramètres de test (basés sur artif_background_parameters.txt)
phi_test = [0.2, 0.3, 0.3, 0.2]  # A, C, G, T
