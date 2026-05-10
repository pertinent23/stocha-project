import numpy as np
import concurrent.futures
import os
from question_222 import load_sequences
from question_223 import run_gibbs_parallele, calcul_score_consensus

def estimer_modele_fond(sequences):
    """
    Estime le modèle de fond (Phi) d'ordre 0 en calculant 
    les fréquences globales des nucléotides dans les séquences.
    """
    comptages = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    total = 0
    for seq in sequences:
        for nuc in seq:
            if nuc in comptages:
                comptages[nuc] += 1
                total += 1
    
    phi = [comptages['A']/total, comptages['C']/total, 
           comptages['G']/total, comptages['T']/total]
    return phi

def generer_fichiers_soumission(positions, theta, W, nom_base):
    """
    Génère les fichiers motifs-*.txt et pwm-*.txt au format standard.
    """
    # 1. Fichier des positions (motifs-*.txt)
    # Format : start,end (indices 0-based)
    with open(f"./results/motifs-{nom_base}.txt", "w") as f:
        for p in positions:
            f.write(f"{p},{p + W - 1}\n")
            
    # 2. Fichier PWM (pwm-*.txt)
    # Format : chaque ligne = une ligne de la matrice theta
    with open(f"./results/pwm-{nom_base}.txt", "w") as f:
        for i in range(4): # Lignes A, C, G, T
            ligne_str = ",".join([str(val) for val in theta[i, :]])
            f.write(ligne_str + "\n")
            
    print(f"Fichiers motifs-{nom_base}.txt et pwm-{nom_base}.txt générés avec succès !")

def resoudre_competition(fichier_seq, w_min, w_max, nom_base, num_restarts=10, iterations=500):
    """
    Teste toutes les longueurs W possibles et garde celle avec la plus forte 
    densité d'information par colonne.
    """
    print(f"\n{'='*50}")
    print(f"RÉSOLUTION POUR {nom_base.upper()} (W entre {w_min} et {w_max})")
    print(f"{'='*50}")
    
    sequences = load_sequences(fichier_seq)
    phi_estime = estimer_modele_fond(sequences)
    print(f"Modèle de fond estimé (A,C,G,T) : {[round(p, 3) for p in phi_estime]}")
    
    meilleur_W = w_min
    meilleur_score_moyen = -1
    meilleures_pos = []
    meilleur_theta = None
    
    for W in range(w_min, w_max + 1):
        print(f"\n--- Évaluation pour W = {W} ---")
        pos, theta = run_gibbs_parallele(fichier_seq, W, phi_estime, num_restarts, iterations)
        
        # Calcul du score de consensus et sa densité par colonne
        # Le score total augmente naturellement avec W, donc on le normalise
        # en divisant par le nombre de colonnes pour une comparaison équitable
        score_total = calcul_score_consensus(theta)
        score_moyen = score_total / W
        
        print(f"-> Consensus: {score_total:.2f} bits | Densité (bits/col) : {score_moyen:.4f}")
        
        if score_moyen > meilleur_score_moyen:
            meilleur_score_moyen = score_moyen
            meilleur_W = W
            meilleures_pos = pos
            meilleur_theta = theta
            
    print(f"\n>>> MEILLEURE LONGUEUR TROUVÉE : W = {meilleur_W} (Densité : {meilleur_score_moyen:.4f} bits/col)")
    
    # Génération des fichiers finaux
    generer_fichiers_soumission(meilleures_pos, meilleur_theta, meilleur_W, nom_base)
    return meilleur_W

# =====================================================================
# BLOC PRINCIPAL D'EXÉCUTION
# =====================================================================
if __name__ == '__main__':
    
    # Résolution pour les séquences artificielles (W dans [10, 15])
    fichier_artif = "./../sequences-competition/sequences-artif-competition.txt"
    if os.path.exists(fichier_artif):
        resoudre_competition(fichier_artif, 10, 15, "artif", num_restarts=15, iterations=600)
    else:
        print(f"Fichier {fichier_artif} introuvable.")

    # Résolution pour les séquences réelles (W dans [15, 30])
    # Note : Le calcul peut être long (testez avec num_restarts réduit si nécessaire)
    fichier_reel = "./../sequences-competition/sequences-reel-competition.txt"
    if os.path.exists(fichier_reel):
        resoudre_competition(fichier_reel, 15, 30, "reel", num_restarts=10, iterations=500)
    else:
        print(f"Fichier {fichier_reel} introuvable.")