HAC (clustering hiérarchique agglomératif)
1) Calculer matrice des distances d(i,j).
2) Trouver la plus petite distance -> fusionner les 2 clusters.
3) Mettre à jour la matrice avec une règle de linkage.
4) Répéter jusqu’à 1 cluster -> dendrogramme (hauteur = distance de fusion).

Linkage (distance entre clusters C et D):
- SINGLE (min): d(C,D)=min_{x∈C,y∈D} d(x,y).
- COMPLETE (max): d(C,D)=max_{x∈C,y∈D} d(x,y).

Mise à jour après fusion:
Si on fusionne U = (A ∪ B), alors pour tout autre cluster K:
- Single: d(U,K) = min(d(A,K), d(B,K))
- Complete: d(U,K) = max(d(A,K), d(B,K))
(les d(A,K) sont les distances cluster-cluster déjà dans la matrice à l’étape courante)

Exercice a,b,c,d,e (distances clés):
d(c,d)=11 ; d(d,e)=17 ; d(b,c)=21 ; d(a,b)=23 ; d(c,e)=25 ; ...

Single: fusions -> (c,d)@11 ; (cd,e)@17 ; (cde,b)@21 ; (bcde,a)@23
Complete: fusions -> (c,d)@11 ; (a,b)@23 ; (cd,e)@25 ; (ab,cde)@50