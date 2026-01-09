COMMENT REMPLIR LE TABLEAU (PERCEPTRON)

Donné: x0=1, poids courants (w0,w1,w2).
Pour chaque exemple (x0,x1,x2,c) dans l’ordre:

1) Mettre dans la ligne: Input = (x0,x1,x2) et c (cible).
2) Calculer s = Σ_i wi*xi = w0*x0 + w1*x1 + w2*x2.
   -> Remplir la colonne "Σ wi xi" avec s.
3) Calculer o avec Heaviside:
   o=1 si s>0, sinon o=0.
   -> Remplir la colonne "o".
4) Calculer erreur Δ = (c - o).
5) Mise à jour des poids (à écrire dans les colonnes w0,w1,w2 de la fin de ligne):
   w0_new = w0 + Δ*x0
   w1_new = w1 + Δ*x1
   w2_new = w2 + Δ*x2  
6) Ligne suivante: reprendre w_new comme nouveaux poids.

Stop: quand un passage complet sur tous les exemples ne modifie plus les poids.
