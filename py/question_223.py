import numpy as np
import concurrent.futures
from question_222 import gibbs_motif_discovery, phi_test


def calcul_score_consensus(theta):
    """
    Calcule le contenu en information (Consensus Score)
    d'une matrice PWM (Theta).
    Formule : Somme sur les colonnes de ( 2 + Somme(p_i * log2(p_i)) )
    """
    score_total = 0
    # On ajoute un epsilon très petit pour éviter log2(0)
    eps = 1e-10
    theta_safe = np.maximum(theta, eps)

    for j in range(theta.shape[1]):
        colonne = theta_safe[:, j]
        # Entropie de Shannon
        entropie = -np.sum(colonne * np.log2(colonne))
        # Contenu en information (2 bits max pour l'ADN)
        information_content = 2.0 - entropie
        score_total += information_content

    return score_total


def calcul_jaccard_index(vrai_debut, pred_debut, W):
    """
    Calcule l'indice de Jaccard entre deux fenêtres de même taille W.
    """
    # Calcul du chevauchement (intersection)
    intersection = max(0, W - abs(vrai_debut - pred_debut))

    if intersection == 0:
        return 0.0

    # Calcul de l'union (taille totale couverte par les deux fenêtres)
    union = 2 * W - intersection
    return intersection / union


def calcul_aji(vraies_positions, pred_positions, W):
    """
    Calcule l'Average Jaccard Index (AJI) sur l'ensemble des séquences.
    """
    jaccards = []
    for v, p in zip(vraies_positions, pred_positions):
        jaccards.append(calcul_jaccard_index(v, p, W))
    return np.mean(jaccards)


def charger_positions_depuis_fichier(filepath):
    """
    Charge les vraies positions à partir du fichier texte.
    Ignore les lignes d'en-tête (qui commencent par '>').
    """
    positions = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.startswith(">"):
                positions.append(int(line))
    return positions


def charger_positions_csv(filepath):
    """
    Charge les positions de début depuis un fichier CSV (motifs-purr.txt).
    Retourne une liste d'entiers correspondant à la colonne 'start'.
    Ignore la première ligne d'en-tête.
    """
    positions = []
    with open(filepath, "r") as f:
        next(f)  # saute l'en-tête
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                positions.append(int(parts[1]))
    return positions


def _executer_un_gibbs(args):
    """
    Wrapper pour le multiprocessing.
    Exécute une instance indépendante de l'algorithme de Gibbs et retourne son score.
    """
    seq_file, W, phi, iterations = args
    pos, theta = gibbs_motif_discovery(seq_file, W, phi, iterations)
    score = calcul_score_consensus(theta)
    return score, pos, theta


def run_gibbs_parallele(seq_file, W, phi, num_restarts=10, iterations=500):
    """
    Lance l'échantillonneur de Gibbs plusieurs fois en parallèle.
    Retourne la solution avec le meilleur score de consensus.
    
    Le multiprocessing permet de réduire le temps de calcul :
    chaque restart indépendant s'exécute sur un cœur différent du CPU.
    """
    meilleur_score = -1
    meilleures_positions = []
    meilleur_theta = None

    print(
        f"\n[DÉMARRAGE] {num_restarts} "
        f"exécutions parallèles sur {seq_file}..."
    )

    # On prépare les arguments pour chaque processus (un tuple par exécution)
    taches = [(seq_file, W, phi, iterations) for _ in range(num_restarts)]

    # Utilisation de ProcessPoolExecutor pour distribuer sur les coeurs du CPU
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map exécute _executer_un_gibbs pour chaque tuple dans taches
        resultats = executor.map(_executer_un_gibbs, taches)

        # On analyse les résultats au fur et à mesure qu'ils reviennent
        for i, (score, pos, theta) in enumerate(resultats):
            print(
                f" Exécution {i+1}/{num_restarts} "
                f"terminée -> Score : {score:.2f} bits"
            )

            # On conserve strictement la meilleure chaîne de Markov
            if score > meilleur_score:
                meilleur_score = score
                meilleures_positions = pos
                meilleur_theta = theta

    print(f"[SUCCÈS] Meilleur score retenu : {meilleur_score:.2f} bits !")
    return meilleures_positions, meilleur_theta


if __name__ == '__main__':

    # ---------------------------------------------------------
    # 1. ÉVALUATION SUR DONNÉES ARTIFICIELLES
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ÉVALUATION SUR DONNÉES ARTIFICIELLES")
    print("="*50)

    W_artif = 10
    vraies_pos_artif = charger_positions_depuis_fichier(
        "../sequences-artif/artif_start_positions.txt"
    )

    # Utilisation de la nouvelle fonction parallèle (10 redémarrages)
    pos_trouvees_artif, theta_trouve_artif = run_gibbs_parallele(
        '../sequences-artif/sequences_artif.txt',
        W=W_artif,
        phi=phi_test,
        num_restarts=10,
        iterations=500  # 500 itérations suffisent généralement avec le restart
    )

    score_aji_artif = calcul_aji(vraies_pos_artif, pos_trouvees_artif, W_artif)
    print(f"\n---> Average Jaccard Index (AJI) final : {score_aji_artif:.4f}")

    score_cons_artif = calcul_score_consensus(theta_trouve_artif)
    print(
        f"---> Score de consensus final : "
        f"{score_cons_artif:.2f} / {W_artif * 2} bits"
    )

    # ---------------------------------------------------------
    # 2. ÉVALUATION SUR DONNÉES BIOLOGIQUES (PURR)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("ÉVALUATION SUR DONNÉES BIOLOGIQUES (PURR)")
    print("="*50)

    W_purr = 16
    vraies_pos_purr = charger_positions_csv(
        "../sequences-purr/motifs-purr.txt"
    )

    # Attention: phi_test ici est utilisé pour PurR, dans l'idéal il faudrait
    # le recalculer empiriquement pour le fichier PurR,
    # mais on garde votre logique.
    pos_trouvees_purr, theta_trouve_purr = run_gibbs_parallele(
        '../sequences-purr/sequences-purr.txt',
        W=W_purr,
        phi=phi_test,
        num_restarts=10,
        iterations=500
    )

    score_aji_purr = calcul_aji(vraies_pos_purr, pos_trouvees_purr, W_purr)
    print(f"\n---> Average Jaccard Index (AJI) final : {score_aji_purr:.4f}")

    score_cons_purr = calcul_score_consensus(theta_trouve_purr)
    print(
        f"---> Score de consensus final : "
        f"{score_cons_purr:.2f} / {W_purr * 2} bits"
    )
