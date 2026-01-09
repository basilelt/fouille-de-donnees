But:
- Partitionner les données en k groupes (clusters) non chevauchants.
- Objectif: minimiser la variance intra-cluster (points proches du centroïde).

Notations:
- Données: points xi (souvent en 2D dans les exos).
- k = nombre de clusters.
- Centroïdes: μ1,...,μk (centres des clusters).

Algorithme (boucle):
1) Choisir k.
2) Initialiser: choisir k points/valeurs comme centres (centroïdes initiaux).
3) ASSIGN (affectation):
   Pour chaque point x, calculer sa distance à chaque μj et l’affecter au cluster le plus proche.
4) UPDATE (mise à jour):
   Recalculer chaque centroïde μj = moyenne (mean) de tous les points affectés au cluster j.
5) Répéter ASSIGN + UPDATE jusqu’au critère d’arrêt:
   - plus aucun point ne change de cluster, ou nb d’itérations max, etc.

Distance (cas le plus courant en exo):
- Distance euclidienne en 2D: d((x,y),(a,b)) = sqrt((x-a)^2 + (y-b)^2).
(Astuce calcul: comparer d^2 pour éviter la racine.)

À rendre souvent dans un exercice:
- Tableau des distances point->centroïdes.
- Affectation finale de chaque point.
- Centroïdes finaux (moyennes).

=== EXERCICE DU PDF (clients e-commerce) ===
Données (visites, achats):=== K-MEANS (CLUSTERING) ===

But:
- Partitionner les données en k groupes (clusters) non chevauchants.
- Objectif: minimiser la variance intra-cluster (points proches du centroïde).

Notations:
- Données: points xi (souvent en 2D dans les exos).
- k = nombre de clusters.
- Centroïdes: μ1,...,μk (centres des clusters).

Algorithme (boucle):
1) Choisir k.
2) Initialiser: choisir k points/valeurs comme centres (centroïdes initiaux).
3) ASSIGN (affectation):
   Pour chaque point x, calculer sa distance à chaque μj et l’affecter au cluster le plus proche.
4) UPDATE (mise à jour):
   Recalculer chaque centroïde μj = moyenne (mean) de tous les points affectés au cluster j.
5) Répéter ASSIGN + UPDATE jusqu’au critère d’arrêt:
   - plus aucun point ne change de cluster, ou nb d’itérations max, etc.

Distance (cas le plus courant en exo):
- Distance euclidienne en 2D: d((x,y),(a,b)) = sqrt((x-a)^2 + (y-b)^2).
(Astuce calcul: comparer d^2 pour éviter la racine.)

À rendre souvent dans un exercice:
- Tableau des distances point->centroïdes.
- Affectation finale de chaque point.
- Centroïdes finaux (moyennes).

=== EXERCICE DU PDF (clients e-commerce) ===
Données (visites, achats):
c1(2,1) c2(1,1) c3(2,2) c4(1,2) c5(6,6) c6(5,5) c7(5,6) c8(5,3) c9(5,2) c10(4,2)

Centres initiaux:
C1: (1.5, 3.0)
C2: (4.0, 0.5)
C3: (2.5, 5.0)

Après convergence (centres/means indiqués dans le PDF):
μ1 = (1.50, 1.50)
μ2 = (4.66, 2.33)
μ3 = (5.33, 5.66)

Problèmes classiques:
- Sensible à l’initialisation (résultats différents selon les centres initiaux).
- Sensible aux outliers (un point extrême peut “tirer” un centroïde).
Amélioration citée: k-means++ pour mieux initialiser.

c1(2,1) c2(1,1) c3(2,2) c4(1,2) c5(6,6) c6(5,5) c7(5,6) c8(5,3) c9(5,2) c10(4,2)

Centres initiaux:
C1: (1.5, 3.0)
C2: (4.0, 0.5)
C3: (2.5, 5.0)

Après convergence (centres/means indiqués dans le PDF):
μ1 = (1.50, 1.50)
μ2 = (4.66, 2.33)
μ3 = (5.33, 5.66)

Problèmes classiques:
- Sensible à l’initialisation (résultats différents selon les centres initiaux).
- Sensible aux outliers (un point extrême peut “tirer” un centroïde).
Amélioration citée: k-means++ pour mieux initialiser.