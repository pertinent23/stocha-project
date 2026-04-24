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
    ÉTAPE A : Tire un nouveau profil de motif (Theta) sachant les positions.

    Utilise la conjugaison de Dirichlet (Prior) et Multinomiale (Vraisemblance)
    pour échantillonner une nouvelle matrice de probabilités.

    Args:
        counts (np.ndarray): Matrice des comptages (4 x W).
        alpha (float, optional): Pseudo-compte (Prior). Défaut à 1.0.

    Returns:
        np.ndarray: Nouvelle matrice Theta des probabilités du motif.
    """
    W = counts.shape[1]
    theta = np.zeros((4, W))
    for j in range(W):
        # Tirage Dirichlet : Dirichlet(alpha + n_ij)
        theta[:, j] = np.random.dirichlet(counts[:, j] + alpha)
    return theta


def sample_positions(sequences, theta, phi, W):
    """
    ÉTAPE B : Tire de nouvelles positions sachant le profil (Theta).

    Calcule le ratio (Motif / Bruit de fond) pour chaque position possible,
    le normalise en probabilités réelles, puis effectue un tirage.

    Args:
        sequences (list): Liste des séquences.
        theta (np.ndarray): Matrice de probabilités du motif actuel.
        phi (list): Probabilités du bruit de fond (A, C, G, T).
        W (int): Longueur du motif.

    Returns:
        list: Nouvelles positions de départ pour le motif.
    """
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    new_positions = []

    for seq in sequences:
        L = len(seq)
        possible_starts = L - W + 1
        weights = np.zeros(possible_starts)

        for a in range(possible_starts):
            # Calcul du poids w_a (Ratio Motif / Bruit de fond)
            w = 1.0
            for j in range(W):
                nuc = seq[a + j]
                idx = nuc_to_idx[nuc]
                w *= (theta[idx, j] / phi[idx])
            weights[a] = w

        # Normalisation pour obtenir une distribution de probabilité
        prob = weights / np.sum(weights)

        # Tirage de la position (Méthode de la transformée inverse via numpy)
        choix = np.random.choice(possible_starts, p=prob)
        new_positions.append(choix)

    return new_positions


def phase_shift_move(sequences, positions, theta, phi, W):
    """
    Tente de décaler toutes les positions d'un coup (Metropolis-Hastings).

    C'est une astuce pour éviter que l'algorithme ne reste bloqué
    dans un minimum local (décalage de phase).

    Args:
        sequences (list): Liste des séquences.
        positions (list): Positions actuelles.
        theta (np.ndarray): Matrice de profil actuelle.
        phi (list): Probabilités de fond.
        W (int): Longueur du motif.

    Returns:
        list: Les nouvelles positions (ou les anciennes si le saut est rejeté).
    """
    shift = np.random.choice([-1, 1])
    new_pos = [p + shift for p in positions]

    # Vérifier si le décalage sort des limites des séquences
    for i, p in enumerate(new_pos):
        if p < 0 or p + W > len(sequences[i]):
            return positions

    if np.random.rand() < 0.3:
        return new_pos
    return positions


def gibbs_motif_discovery(seq_file, W, phi, iterations=1000):
    """
    L'algorithme principal de l'échantillonneur de Gibbs.

    Boucle itérative alternant entre l'échantillonnage du profil (Theta)
    et l'échantillonnage des positions.

    Args:
        seq_file (str): Nom du fichier contenant les séquences.
        W (int): Longueur du motif recherché.
        phi (list): Probabilités du bruit de fond.
        iterations (int, optional): Nombre d'itérations. Défaut à 1000.

    Returns:
        tuple: (positions_finales, matrice_theta_finale)
    """
    sequences = load_sequences(seq_file)

    # Initialisation : Positions choisies totalement au hasard
    positions = [np.random.randint(0, len(s) - W + 1) for s in sequences]

    for i in range(iterations):
        # 1. Échantillonner le profil (Étape A)
        counts = get_counts(sequences, positions, W)
        theta = sample_theta(counts)

        # 2. Échantillonner les positions (Étape B)
        positions = sample_positions(sequences, theta, phi, W)

        # 3. Mouvement de décalage de phase (toutes les 10 itérations)
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

# final_pos, final_theta = gibbs_motif_discovery(
#     '../sequences-artif/sequences_artif.txt', 10, phi_test
# )
# print("Positions finales des motifs :", final_pos)
# print("Matrice Theta finale :\n", final_theta)
# print("\n")
