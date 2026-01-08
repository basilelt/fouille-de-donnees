# Auto-generated content module for Cours
# Contains embedded markdown content for this topic

TOPIC_NAME = "Cours"
TOPIC_KEY = "cours"

CONTENT = {
    "cours/cours.md": """# Synthèse Data Mining - Cours Germain Forestier

## 1. Introduction au Data Mining
Le Data Mining (fouille de données) découvre modèles, relations et connaissances dans grands ensembles de données. Combine statistiques, IA et Machine Learning.

### Processus CRISP-DM
1. **Business Understanding**: Définir objectifs métier et critères de succès.
2. **Data Understanding**: Collecter, décrire et explorer données (qualité).
3. **Data Preparation**: Nettoyage, fusion, transformation (80% du temps).
4. **Modeling**: Sélectionner et appliquer algorithmes.
5. **Evaluation**: Vérifier si modèle répond aux objectifs business.
6. **Deployment**: Mise en production et maintenance.

### Prétraitement et Qualité des Données
- **Structurées**: Tables (CSV, SQL). **Non structurées**: Texte, images, vidéos.
- **Nettoyage**: Gérer valeurs manquantes (suppression ou imputation par moyenne/médiane/mode), détecter outliers.
- **Transformation**: Normalisation Min-Max $[0, 1]$ ou Z-score (moy 0, écart-type 1).
- **EDA**: Visualisation (Histogrammes: distribution, Boxplots: dispersion/outliers, Nuages: relations) pour comprendre données.

### Concepts Statistiques et ML
- **Supervisé**: Données étiquetées (Classification: discret, Régression: continu).
- **Applis**: Contrôle qualité, prédiction pannes, météo, rendements.

### Data Analysis vs. Data Mining vs. Machine Learning
- **Data Analysis**: Compréhension données via méthodes statistiques.
- **Data Mining**: Utilise algos pour trouver modèles/prédire tendances. Inclut prétraitement, EDA, interprétation.
- **Machine Learning (ML)**: Sous-ensemble IA, entraîne modèles. Utilisé dans Data Mining.

### Intersections
- IA, ML, Big Data, Rech. Info, Reco. Formes.
- Text Mining, BDD, Anal. Prédictive, NLP, Vision (OCR).

### Relation AI, ML, Deep Learning
Deep Learning $\\subset$ Machine Learning $\\subset$ Intelligence Artificielle.

### "Tendance" et Data Scientist
- Termes ML, Big Data, Data Science, AI souvent interchangeables.
- Data Scientist: Profil transdisciplinaire très demandé (AI engineer/scientist).

## 2. Types et Sources de Données
### Structurées vs. Non Structurées
- **Structurées**: Format défini (tables, CSV). Ex: Iris.
- **Non Structurées**: Pas de forme spécifique (texte, vidéos, images). Traitement spécialisé.
- **Problème**: Classes mal balancées (ex: maladies rares).

### Formats Courants
- **CSV**: Texte brut, sép. virgules.
- **JSON**: Texte, lisible, web.
- **XML**: Balisage, auto-descriptif.
- **Autres**: YAML, HDF5.

### Sources
- **BDD**: Structurées (relationnelles, doc, clé-valeur).
- **Web Scraping**: Extraction sites web.
- **APIs**: Protocoles interaction apps/données.

## 3. Prétraitement des Données
### Données Propres et Qualité
- **Propres**: Sans erreurs/incohérences, améliore précision, facilite intégration.
- **Qualité**: Normes (précision, complétude, fiabilité), soutient décision.
- **Mauvaise qualité**: Conclusions incorrectes, coûts up, confiance down.
- **Note**: Prép = bcp de temps. "Cleaned dataset" nécessaire.

### Techniques Nettoyage
- **Val. manquantes**: Supprimer, Imputer (moy, med, mode), Prédire (ML).
- **Transformation**: Norm/scale, conversion types, encodage catégoriel.
- **Outliers**: Identifier/gérer. Prudence (info importante). Pertinents dans train/test.
- **Validation**: Vérifier précision avec sources externes.

## 4. Analyse Exploratoire (EDA)
### But
- **Comprendre struct**: Forme, tendance centrale, dispersion, relations.
- **Identifier modèles**: Tendances, clusters, outliers.
- **Communiquer**: Visualisations.

### Visualisation
- **Barres**: Fréq/catégorie.
- **Nuages points**: Relation 2 vars continues.
- **Linéaires**: Tendances temps.
- **Histogrammes**: Distrib var continue.
- **Boxplots**: Tend centrale/variabilité.

## 5. Outils et Logiciels
### Populaires
- **Python**: Généraliste, écosystème riche (pandas, matplotlib, scikit-learn, NumPy).
- **R**: Calcul stat.
- **SQL**: Requêtes BDD.
- **Excel**: Tableur.

### Bibliothèques
- **Pandas**: Manip/analyse (dataframe).
- **NumPy**: Calcul num, matrices.
- **Matplotlib**: Visu.
- **Scikit-learn**: ML (classif, reg, clust, preproc).

## 6. Concepts Base Stat et ML
### Descriptives vs. Inférentielles
- **Descriptives**: Résumé (moy, med, mode, écart-type). Aperçu.
- **Inférentielles**: Préd/inférences sur pop via échantillon. Conclusions générales.

### Supervisé vs. Non Supervisé
- **Supervisé**: Données étiquetées. Apprend fct entrée->sortie (classif, reg). Prédire futur.
- **Non Supervisé**: Non étiquetées. Ident modèles/struct (clustering, réduction dim). Description.

### Biais-Variance
- **Biais**: Erreur simplif excessive -> sous-apprentissage (underfitting).
- **Variance**: Sensibilité fluctuations -> sur-apprentissage (overfitting).
- **Compromis**: Équilibrer pour min erreur totale. Précision et interprétabilité.

## 7. Processus CRISP-DM
1. **Business Understanding**: Obj métier. Alignement, métriques, ressources.
2. **Data Understanding**: Collecte, descr, explo, qualité.
3. **Data Preparation**: Clean, transform, integrate, select.
4. **Modeling**: Sélec techniques, build, eval.
5. **Evaluation**: Qualité modèle, révision.
6. **Deployment**: Plan déploiement, maint, surv.

## 8. Big Data et Scalabilité
### 5 V
- **Volume**: Taille (To, Po).
- **Vélocité**: Vitesse gén/traitement.
- **Variété**: Types (struct, non-struct).
- **Véracité**: Qualité/fiabilité.
- **Valeur**: Potentiel dérivé.
- **Tech**: Hadoop, Spark, NoSQL.

### Défis
- Stockage, Traitement (//), Intégration, Qualité, Sécurité, Analyse, Scalabilité, Coût.
- **Pb**: Algos durs sur énormes datasets -> sous-ensembles.

## 9. Exemples Concrets
### Applications
- **Santé**: Épidémies, traitements.
- **Finance**: Fraude, risques.
- **Retail**: Recom, prix, stocks (Walmart).
- **Fab**: Qualité, process.
- **Transport**: Trafic, itin (GMap), maint (GE).
- **Énergie**: Prév demande.
- **Divert**: Recom (Netflix).
- **Gouv**: Séc publique, météo (NWS).

### Succès
Netflix, AmEx, Walmart, GE Aviation, Google Maps, Watson Santé, NWS, LinkedIn.

# Bayes
## Résumé
Classifieur proba (Th. Bayes + indép attributs). Trouve classe la plus probable. Utile classif/texte.

## 1. Intro
- Utilise probas pour classif.
- Affecte proba à chaque hypothèse (classe).
- Obs train modifient distrib proba.
- Cherche hyp la plus probable sachant instance.
- Base: Proba cond + Bayes. Hypothèse indép attributs.

## 2. Rappels Proba
- P(A) $\\in [0,1]$.
- Indép: $P(A \\cap B) = P(A)P(B)$.
- Cond: $P(A|B) = P(A \\cap B) / P(B)$.

## 3. Th. Bayes
$$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$

## 4. Application Classification
- Calc $P(C_k|X)$ pour chaque classe.
- $P(C|Desc) = \\frac{P(Desc|C)P(C)}{P(Desc)}$
- **Postériori**: $P(C|Desc)$. **Vraisemblance**: $P(Desc|C)$. **Priori**: $P(C)$.
- Estim via train: $P(C)$=prop classe, $P(Desc|C)$=freq desc ds classe.

### 4.2 MAP (Maximum A Posteriori)
- Choisir $C_k$ max $P(C_k|Desc)$.
- $P(Desc)$ constant -> $\\text{argmax}_k [P(Desc|C_k)P(C_k)]$.

## 5. Naive Bayes (Indép)
- Hyp: Attributs indép sachant classe.
- $P(Desc|C) = \\prod P(a_i|C)$.
- **Final**: $\\text{argmax}_k [P(C_k) \\times \\prod P(a_i|C_k)]$.
- Discret: $P(a_i|C_k)$ = freq val $a_i$ ds $C_k$.

## 6. Ex: Jeu
- Train: Genre, Plat, Budg, Pop(O/N). X: RPG, PC, Med.
- **Yes**: $P(Y)=3/5$. Prod $P(attr|Y)=0.0444$.
- **No**: $P(N)=2/5$. Prod $P(attr|N)=0.05$.
- **Pred**: No ($0.05 > 0.0444$).
- **Norm**: Somme=0.0944. P(Y)=47%, P(N)=53%.

## 7. Petites Probas
- Prod $< 1$ -> valeurs très petites.
- **Log**: $\\log(ab)=\\log a+\\log b$. Somme évite underflow.

## 8. Numérique
- Estimer distrib (Gaussienne: $\\mu, \\sigma$ par classe).

## 9. Analyse Texte
- Classif (spam).
- **Bag-of-Words**: Mot=attr (présence/fréq).

## 10. Avantages/Inconvénients
- **+**: Simple, efficace, interprétable, perf petits data, scale.
- **-**: Indép attr (rare), Zéro Proba (lissage Laplace), Modèles complexes.

# Clustering
## Intro
- Non supervisé. Groupes (clusters).
- Max similarité intra, Min similarité inter.

## Distance
- Mesure ressemblance.
- **Euclidienne**: $\\sqrt{\\sum \\text{diff}^2}$. **Norm** (Min-Max) cruciale.
- **Matrice Dist**: $2 \\times 2$, complexité quadratique.

## Hiérarchique Ascendant
1. **Init**: Chq obj = clust.
2. **Iter**: Fusion 2 + proches.
3. **Fin**: 1 seul clust.
- **Dendrogramme**: Arbre fusions. Coupe -> nb clusters.
- **Linkage**: Min (Single, chaînage), Max (Complete), Moy (Average), Ward (inertie, équil).

## Eval
- Pas vérité terrain.
- **Interne**: Qual intrinsèque (Silhouette, Dunn).
- **Externe**: Comp existant (Rand, Adj Rand).

## Concl
- **+**: Simple, nb classes flex, visu.
- **-**: $O(n^2)$ lourd, glouton (irrév), sensib lien.

# Decision Tree
## Quoi?
- Arbre: Nœuds (tests), Branches (val), Feuilles (classe).

## Construction (ID3)
- Top-down, iter.
- **Entropie**: Incertitude. **Gain Info**: Réduc incert.
- Choisir attr max gain -> Nœud.

## Numérique
- **Discrétisation**: Seuils (ex: $<70$). C4.5 le fait.

## Surapprentissage (Overfitting)
- Modèle trop collé train.
- **Pruning**: Suppr nœuds, pénalité compl, prof max.

## Random Forests
- Ensemble d'arbres. Vote.
- **Div**: Algos diff, éch data/attr.
- **+**: Perf élevée.

## Pros/Cons
- **+**: Interprétable, peu prép, num/cat, non-lin.
- **-**: Overfit, instable, biais, opt local.

# Kmeans
## Principes
- Non sup, Partition (K), K fixé.
- Clusters compacts. Means = Moyennes.

## Algo
1. Select K.
2. Init K centres.
3. Affect obj centre + proche.
4. Recalc centres (moy).
5. Répéter tant que bouge.

## Aspects
- **K**: Délicat. Intuition, Coude, Silhouette.
- **Dist**: Euclid, Manh.
- **Init**: Sensible (opt local). K-means++.
- **Outliers**: Influencent moy. K-medoids.

## Variantes
- **K-medoids** (obj réel), **Fuzzy** (degré), **K-means++** (init).

## Pros/Cons
- **+**: Rapide, Interpr, Simple.
- **-**: Choix K, Forme sphérique, Sensib init/outliers.

# Knn
## Takeaway
Lazy, classif/reg via maj/moy k voisins. Simple mais lourd.

## Intro
- Instance-based/Lazy. Classif new via maj k voisins. 1-NN: + proche.

## Concepts
- **Dist**: Euclid, Manh.
- **Lazy**: Pas modèle, tout train = modèle.

## Pros/Cons
- **+**: Simple, Polyv, Rapide (no train), Non-param, Update, Explic.
- **-**: Coût ($O(N)$), Bruit (pt k), Choix k (impair), Norm indispensable.

## Détails
- **Ex**: Iris.
- **Pondér**: Vote maj ou dist ($1/d^2$).
- **Cat**: One-Hot.

# Neural Network
## Hist
- 60s: Perceptron. XOR imp -> AI Winter.
- 74: Backprop (Werbos).
- 90s: CNN (LeCun).
- 2012: AlexNet (DL boom).

## Perceptron
- Entrées+Biais -> Poids -> Somme -> Act -> Sortie.
- **Apprent**: Corr err. $W_{new} = W + \\eta(C-O)X$.

## Deep Learning
- Profond, bcp params. Non-lin.
- **Archis**: LeNet-5, AlexNet, Inception, VGG16, Transformers.
- Facteurs: Profondeur, GPU.

# Time Series
## Intro
- Séq ordonnée tps. Ordre compte.
- Uni/Multi. Decomp (Trend, Sais, Cyc, Bruit).

## Feat Extract
- Stats ($\\mu, \\sigma$, Z-Norm). Auto-Corr. Fourier. catch22.

## Analysis
- **Forec**: MA (err), AR (val).
- **Imput**: Moy, Med, Ffill, Bfill.
- **Classif**: KNN (Euclid), DTW (align temp).
- **Unequal**: Pad, Trunc, DTW.
- **Dim Red**: PCA, PAA. Symb: SAX, SFA.

# Wooclap
**DM**: Patterns. **Struct**: CSV. **Preproc**: Clean. **EDA**: Understand. **Miss**: Mean. **Dim**: PCA. **BDD**: SQL. **Unsup**: Clust. **Big**: Insights. **Qual**: Eval. **Unstr**: No form. **Pred**: ML. **Ver**: Qual. **Visu**: Hist.
**KNN**: Inst-based, Euclid, Cost, Noise(sm k), Scale, W-vote, Lazy, OneHot, CV, Med, Norm, Odd k.
**DT**: Class/Reg, Test node, ID3, Entr, Overfit, Prun, CCP, InfoGain, RF, Num/Cat, MDL, Discr.
**Bayes**: Cond, Indep, P(A), P(AB), MAP, Word, ZeroProb, Sm data, Log, Gauss, Spam.
**Hie Clust**: Grps, Unsup, Euclid, Scale, Dendro, Sgl(noise), Ward(bal), Cut, Silh(int), Rand(ext), Cost, Centr.
**KMeans**: Min var, K rand, Mean, K++, Conv, Euclid, Outlier, Elbow, Spher, Medoids, Unsup, Silh.
**NN**: Rosenb, Nonlin, Werbos, Heaviside, LeNet, AlexNet, Incept, VGG, ErrCorr, 1x1, ReLU, DL, Nonlin bnd.
**TS**: Order, ZNorm, Forec, DTW, Trend, Ffill, Four, Var, AR."""
}


def get_files():
    """Return list of files in this topic."""
    return sorted(CONTENT.keys())


def get_content(file_key):
    """Get content for a specific file."""
    return CONTENT.get(file_key, "")


def search(query):
    """Search for query in topic content."""
    results = []
    for file_key, content in CONTENT.items():
        if query.lower() in content.lower():
            results.append(file_key)
    return results
