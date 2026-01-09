Notations:
- Classes: C = {c1,...,ck}
- Instance X = (a1,...,an) = valeurs d'attributs observées

Bayes:
P(c|X) = P(X|c)*P(c) / P(X)
(P(X) identique pour toutes les classes => inutile pour argmax)

Règle de décision (MAP):
hMAP = argmax_c  P(X|c)*P(c)

Naive Bayes (indépendance conditionnelle):
P(X|c) = Π_i P(ai|c)
=> score(c) = P(c) * Π_i P(ai|c)

Estimation par comptage (attribut discret):
P(ai=v | c) = n_ic / n_c
- n_ic = nb d'exemples de classe c ayant (ai=v)
- n_c  = nb d'exemples de classe c

Algorithme (discret):
1) Pour chaque classe c: calculer P(c) = n_c / n_total
2) Pour chaque attribut i: calculer P(ai=v|c) par fréquences
3) Pour une nouvelle instance X: calculer score(c)=P(c)*Π_i P(ai|c)
4) Prédire la classe avec le plus grand score(c)

Normalisation (optionnelle, pour avoir une "proba" qui somme à 1):
P(c|X)_norm = score(c) / Σ_j score(cj)

Truc anti-underflow (beaucoup d'attributs):
logscore(c) = ln(P(c)) + Σ_i ln(P(ai|c))
argmax des scores == argmax des logscores

=== GAUSSIAN NAIVE BAYES (CONTINU) ===
Si xi est numérique:
P(xi|y) = (1 / sqrt(2π*σ_y^2)) * exp( - (xi-μ_y)^2 / (2*σ_y^2) )

=== RAPPEL EXEMPLE (JEU VIDEO DU PDF) ===
X=(RPG, PC, Medium)
score(Yes) = (2/3)*(1/3)*(1/3)*(3/5) = 2/45 ≈ 0.0444
score(No)  = (1/2)*(1/2)*(1/2)*(2/5) = 1/20 = 0.05
=> prédire "No" car score(No) > score(Yes)