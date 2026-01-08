# Auto-generated content module for Part2
# Contains embedded markdown content for this topic

TOPIC_NAME = "Part2"
TOPIC_KEY = "part2"

CONTENT = {
    "part2/part2.md": """# Kmeans

## Principes Fondamentaux de K-means
- Classification Non Supervisée: L'algorithme ne nécessite pas de données labellisées.
- Partitionnement: Divise un jeu de données en K sous-ensembles.
- Paramètre K: Le "K" dans K-means représente le nombre de clusters que l'on souhaite obtenir. Il doit être fixé a priori, contrairement au clustering hiérarchique.
- Objectif: Créer des clusters les plus compacts possible.
- Moyenne (Means): Le terme "means" fait référence au calcul des moyennes pour définir les représentants des clusters.

## Étapes de l'Algorithme K-means
L'algorithme est itératif et vise à améliorer une partition des données.
1. Sélectionner K: Choisir le nombre K de clusters à trouver.
2. Initialisation: Choisir aléatoirement K points dans l'espace des données pour servir de centres initiaux (ou centroïdes) des clusters. Des travaux existent pour des initialisations plus intelligentes que l'aléatoire simple.
3. Affectation des objets: Balayer toutes les données et affecter chaque objet à son cluster dont le centre est le plus proche.
4. Recalcul des centres: Pour chaque cluster, recalculer son nouveau centre en calculant la moyenne des objets qui lui ont été affectés. Pour des vecteurs numériques, c'est simple; pour des données plus complexes (ex: images), cela peut être plus délicat.
5. Répétition: Répéter les étapes 3 et 4 un certain nombre de fois. L'algorithme s'arrête lorsque: Les résultats sont stables (les centres ne bougent plus, ou très peu d'objets changent de cluster). Un nombre maximal d'itérations est atteint (ex: 10 à 15 itérations).

## Exemple Pratique 1: Données Clients (2D)
- Contexte: Analyse de données clients avec "nombre de visites" et "nombre d'achats" (deux dimensions).
- Processus: 1. Initialisation de 3 centres aléatoires. 2. Affectation des clients (points) aux centres les plus proches. 3. Recalcul des centres, qui se déplacent. 4. Nouvelle affectation des clients aux nouveaux centres. 5. Répétition jusqu'à la stabilisation des centres et des affectations.
- Interprétation Sémantique: Après clustering, un expert peut donner du sens aux groupes: Groupe 1: Peu de visites, peu d'achats. Groupe 2: Beaucoup de visites, beaucoup d'achats. Groupe 3 (potentiellement intéressant): Beaucoup de visites, peu d'achats. Ce groupe peut être ciblé pour des promotions afin d'augmenter les achats.

## Exemple Pratique 2: Images Histopathologiques
- Contexte: Analyse d'images médicales (biopsies) pour assister au diagnostic.
- Application: Clustering des pixels dans l'espace RGB (rouge, vert, bleu) en 3 dimensions, sans considérer la position spatiale des pixels.
- Objectif: Regrouper les pixels ayant des couleurs similaires.
- Résultats: Affichage de l'image segmentée pour K=2, 3, ou 4 clusters (couleurs).
- Analyse: L'expert attribue une sémantique aux clusters de couleurs (ex: cellules violettes sont des lymphocytes, cellules roses des macrophages), permettant de dériver des métriques (nombre de cellules, surface occupée) pour l'aide au diagnostic.
- Limitation: Fonctionne bien sur des images très contrastées; moins efficace sur des images avec des nuances de couleurs subtiles.

## Aspects Importants et Considérations
### Choix du Paramètre K
- Délicat: Impacte directement la qualité des clusters.
- Méthodes: Intuition: Basée sur la visualisation ou la connaissance du domaine. Essais multiples: Lancer l'algorithme plusieurs fois avec différentes valeurs de K et étudier les résultats. Critères internes: Méthodes comme le critère du coude (elbow method) ou le score de silhouette. Validation croisée: Tester sur des échantillons de données. Connaissance expert: Demander à un spécialiste du domaine (ex: médecin).

### Choix de la Distance (Métrique)
- Essentiel: Détermine comment la similarité entre objets est calculée.
- Classique: La distance euclidienne est la plus utilisée pour les vecteurs numériques.
- Autres: Distance de Manhattan, distance Cosine, etc. Le choix dépend de la nature des données et des caractéristiques à prendre en compte.
- Exploration: Il est possible de réaliser plusieurs clusterings avec différentes métriques pour observer les variations.

### Critères d'Arrêt des Itérations
- Stabilité des centres: L'algorithme s'arrête lorsque les centres des clusters ne bougent plus significativement.
- Stabilité des affectations: Plus aucun ou très peu d'objets ne changent de cluster.
- Erreur résiduelle: Observer la compacité des clusters et arrêter quand elle n'évolue plus beaucoup.
- Nombre d'itérations fixe: Définir un nombre maximum d'itérations (ex: 10, 15).

### Sensibilité aux Conditions Initiales
- Optimisation Locale: K-means est un algorithme d'optimisation locale.
- Variabilité des Résultats: Une initialisation aléatoire des centres peut conduire à des résultats différents à chaque exécution.
- Solutions: Lancer K-means plusieurs fois et choisir le meilleur résultat. Utiliser des variantes d'initialisation comme K-means++.

### Sensibilité aux Valeurs Aberrantes
- Impact: Les valeurs aberrantes peuvent fortement influencer le calcul des moyennes et déformer les clusters.
- Solutions: Pré-traitement des données (data cleaning). Mise à l'échelle des données. Utiliser des variantes de K-means moins sensibles, comme K-médoïdes.

## Variantes de K-means
Il existe des centaines de variantes de K-means pour adresser ses limitations ou adapter à des besoins spécifiques.
- K-médoïdes (K-Medoids): Au lieu de calculer une moyenne abstraite comme centre, il choisit un objet existant des données du cluster comme représentant (médoïde). Moins sensible aux valeurs aberrantes car le médoïde est une donnée réelle.
- Fuzzy K-means (C-Means Flou): Permet à un objet d'appartenir à plusieurs clusters avec un certain degré d'appartenance (partitionnement "flou" ou "doux"). Contrairement à K-means classique qui fait un partitionnement "dur" (un objet = un cluster).
- K-means++: Améliore la phase d'initialisation des centres. Sélectionne les centres initiaux de manière plus intelligente pour mieux couvrir l'espace de données, ce qui améliore la stabilité et la qualité des clusters finaux.
- Bisecting K-means: Découpe itérativement les clusters trop grands en deux jusqu'à atteindre le nombre K désiré.

## Avantages et Inconvénients de K-means
### Avantages
- Efficacité: Algorithme rapide, surtout sur de grands jeux de données.
- Interprétabilité: Le centre de chaque cluster (la moyenne) fournit un prototype facile à interpréter par un expert.
- Simplicité: Peu de paramètres à définir (K, métrique, nombre d'itérations).

### Inconvénients
- Choix de K: La détermination du nombre optimal de clusters (K) peut être difficile.
- Contrainte de Prototype: Avec la distance euclidienne, K-means tend à former des clusters de forme sphérique. Il peut mal performer sur des clusters de formes arbitraires.
- Sensibilité à l'Initialisation: En raison de son aspect d'optimisation locale, des initialisations défavorables peuvent mener à des résultats sous-optimaux. Il est recommandé de lancer l'algorithme plusieurs fois ou d'utiliser K-means++ pour plus de robustesse.
- Sensibilité aux Aberrations: Les points extrêmes peuvent fausser les centres des clusters.

## Conclusion
K-means est un algorithme de clustering très connu, puissant et largement utilisé. Malgré ses inconvénients, il reste une base solide pour l'analyse de données non supervisée, en particulier grâce à sa simplicité et son efficacité. De nombreuses variantes existent pour pallier ses limitations et l'adapter à des contextes spécifiques.

# Knn

## Main Takeaway
Le k-Plus Proche Voisin (k-NN) est un algorithme d'apprentissage automatique basé sur des instances (lazy learning) qui permet de faire de la classification et de la régression en se basant sur la classe majoritaire ou la moyenne des k objets les plus proches dans l'ensemble d'entraînement. Simple à comprendre et à mettre en œuvre, il est adaptable mais présente des défis majeurs en termes de coût computationnel pour de grands volumes de données et nécessite une préparation rigoureuse des données, notamment la normalisation des caractéristiques et le choix optimal du paramètre k.

## Introduction
- Algorithme basé sur des instances (instance-based learning ou lazy learning).
- Permet de faire de la classification (prédire valeur discrète) et de la régression (prédire valeur continue). L'accent est mis sur la classification.
- Fonctionnement: Classifie un nouvel objet en se basant sur la classe majoritaire au sein des k plus proches voisins.
- k est le nombre de voisins à considérer et est un paramètre de l'algorithme.
- 1-Plus Proche Voisin (1-NN): Cas particulier où on cherche uniquement l'objet le plus proche.
- Le k-NN est une famille d'algorithmes avec de nombreuses versions selon le choix de la distance, de k, de la pondération, etc.

## Concepts Clés
### Mesure de Distance
- Nécessaire pour définir la notion de "proximité".
- Distance Euclidienne: Très classique et largement utilisée pour les vecteurs de valeurs numériques.
- Autres distances: Manhattan, Minkowski, similarité cosinus. Le choix dépend du type de données et des caractéristiques souhaitées.

### Apprentissage Paresseux (Lazy Learning)
- Dit "paresseux" car il n'y a pas d'étape de construction de modèle à l'avance.
- Le "modèle" est constitué par l'intégralité des données d'entraînement.
- Lors de l'inférence (classification d'un nouvel objet), il est comparé à tous les objets du jeu d'entraînement.

## Historique et Applications
- Développé dans les années 1950.
- Formalisé par Cover et Hart en 1967.
- Applications courantes: Vision par ordinateur / Reconnaissance d'images. Systèmes de recommandation. Médecine / Diagnostic médical (facilement explicable). Recherche d'information.
- Implémenté dans l'intégralité des librairies d'apprentissage automatique.

## Avantages
- Simplicité: Facile à comprendre et à expliquer (ex: "chercher le plus proche").
- Polyvalence: Utilisable pour la classification et la régression.
- Rapidité de mise en place: Pas de phase de "training" coûteuse en temps de construction de modèle.
- Non-paramétrique: Ne fait pas d'hypothèses sur la distribution des données, peut construire des frontières de classe complexes.
- Mise à jour facile: Les nouvelles données sont simplement ajoutées au jeu d'entraînement et prises en compte instantanément pour les classifications futures, sans reconstruire de modèle.
- Explicabilité: Facile d'expliquer une prédiction en montrant les voisins les plus proches (particulièrement utile en diagnostic médical).

## Inconvénients et Défis
### Coût Algorithmique Élevé (Temps et Mémoire)
- Problématique pour les gros volumes de données ou un grand nombre de caractéristiques.
- Chaque classification implique la recherche des k plus proches voisins dans tout l'ensemble d'entraînement.
- Le calcul de distance peut être coûteux pour des données complexes (images, texte).

### Sensibilité au Bruit
- Un point bruité ou outlier peut fortement influencer la classification des objets à proximité.
- Une petite valeur de k augmente cette sensibilité.

### Choix de k Difficile
- La valeur de k a un impact significatif sur les performances.
- Petit k: Sensible au bruit, risque de surapprentissage (overfitting), frontières de classification complexes.
- Grand k: Plus résistant au bruit, risque de sous-apprentissage (underfitting), frontières de décision plus lisses.
- Méthodes pour choisir k: Essai-erreur, validation croisée (analyse du taux d'erreur).
- Pour les données binaires, choisir un k impair (ex: 3 ou 5) aide à éviter les égalités de vote.

### Nécessité de Normalisation des Caractéristiques
- Indispensable pour que toutes les caractéristiques aient le même poids dans le calcul de distance.

## Exemple Visuel (Données IRIS)
- Utilisation des données IRIS: 150 objets (50 Setosa, 50 Vericolor, 50 Virginica).
- Un jeu d'entraînement et un jeu de test sont définis.
- Pour chaque objet du jeu de test, on cherche le plus proche voisin (ici, en 1-NN) dans le jeu d'entraînement pour assigner sa classe.

## Implémentation Naïve et Optimisations
- Fonction de classification naïve (1-NN): Pour chaque objet_test à classer: Trouver dans l'ensemble d'entraînement (train) l'objet_train le plus proche. Assigner la classe de l'objet_train trouvé à l'objet_test.
- Optimisations: Techniques pour accélérer le processus de recherche des voisins (ex: calcul de lower bound, pruning, mise en cache de mesures de distance).

## Pondération des Voisins
- Version de base: Vote à la majorité simple (tous les voisins ont le même poids).
- Alternative (pondération par distance): Attribuer un poids proportionnel à la distance de chaque voisin. Les voisins plus proches ont un poids plus important. Utile pour résoudre les égalités de vote et donner plus de pertinence aux objets très proches. Exemple de formule: (1/distance^2) (ou d'autres formules).

## Normalisation des Données (Mise à l'Échelle)
- Principe fondamental: Systématiquement mettre à l'échelle ou normaliser les données avant tout calcul de distance.
- Objectif: Ramener les caractéristiques à une même échelle (ex: entre 0 et 1).
- Exemple: La normalisation Min-Max est courante: Valeur_normalisée = (Valeur_originale - Min_caractéristique) / (Max_caractéristique - Min_caractéristique)
- Raison: Sans normalisation, les caractéristiques avec de plus grandes échelles (ex: prix vs. taille en mètres carrés) domineraient le calcul de distance, indépendamment de leur importance réelle.

## Données Catégorielles
- Problème: Les mesures de distance standards sont conçues pour des valeurs numériques.
- Solution: One-Hot Encoding (encodage binaire). Convertit une caractéristique catégorielle en un vecteur numérique binaire. Ex: Une caractéristique "Couleur" avec les valeurs "red", "green", "blue" serait encodée en: "red" -> [1, 0, 0], "green" -> [0, 1, 0], "blue" -> [0, 0, 1]
- Inconvénient: Peut augmenter considérablement le nombre de colonnes si une caractéristique prend beaucoup de valeurs possibles, ce qui impacte la complexité.

## Résumé et Points Clés
- Algorithme adaptable pour classification, régression, recommandation.
- Puissant et simple à expliquer, mais demande un travail significatif sur les données.
- Le choix de la mesure de distance est crucial et dépend du type de données.
- La complexité temporelle est le défi majeur avec de grands datasets.
- Le k-NN est une famille d'algorithmes avec de nombreuses variantes.
- Importance du prétraitement des caractéristiques: mise à l'échelle (normalisation) et gestion des données manquantes.
- Le paramètre k est central et doit être réglé avec soin.
- Avantages notoires: Explicabilité (ex: diagnostic médical) et facilité de mise à jour des données sans reconstruction du modèle.

# Neural Network

## Historique des Réseaux de Neurones
- Années 1960: Rosenblatt propose le Perceptron, un classifieur binaire. Minsky et Papert mettent en évidence l'incapacité des Perceptrons simples à résoudre des problèmes non linéaires (ex: problème XOR). "AI Winter": Période de perte de confiance due à cette limitation.
- 1974: Werbos propose l'algorithme de Rétropropagation du gradient (Backpropagation), permettant de résoudre les problèmes non linéaires en utilisant des réseaux multicouches. L'algorithme est réutilisé et popularisé par Mel, Hinton et Williams pour l'entraînement des réseaux multicouches.
- Années 1990: Apparition des CNN (Convolutional Neural Networks) pour la classification d'images, proposés par Yann LeCun. Yann LeCun reçoit le Prix Turing (équivalent Nobel en informatique) pour ses travaux sur les CNN.

## Le Perceptron : Bases et Fonctionnement
- Définition: Un neurone est une unité de calcul qui prend des entrées, effectue un calcul et renvoie une sortie.
- Entrées: n entrées (x1 à xn) et un biais x0 toujours égal à 1.
- Paramètres: Les poids (weights) W0 à WN sont les valeurs apprises pendant l'entraînement.
- Calcul de la sortie (version simple): 1. Calcul d'une somme pondérée des entrées et des poids (∑(xi * Wi)). 2. Cette somme est passée à une fonction d'activation qui calcule la sortie finale du neurone.
- Limitation: Le perceptron de base modélise des décisions linéaires et ne peut pas résoudre des problèmes non linéaires.
- Représentation schématique: Entrées (x0...xn) -> Poids (W0...Wn) -> Somme pondérée -> Fonction d'activation -> Sortie.

## Algorithme d'Apprentissage du Perceptron : Correction par Erreur
- Principe: Ajuster les poids du réseau pour maximiser le taux de bonnes réponses.
- Étapes: 1. Initialiser le perceptron et ses poids à des valeurs arbitraires. 2. Pour chaque exemple d'entraînement: Présenter l'exemple au réseau. Calculer la sortie. Comparer la sortie avec la classe attendue. Si la classification est incorrecte, ajuster les poids. 3. L'algorithme s'arrête lorsque tous les exemples sont correctement classés et qu'aucun changement de poids n'est nécessaire (stabilité).
- Formule de mise à jour des poids: W_nouveau = W_précédent + η × (C - O) × X_entrée
- η: Taux d'apprentissage (non explicitement mentionné mais implicite dans le coefficient multiplicateur, souvent appelé alpha).
- C: Classe attendue (cible).
- O: Sortie calculée par le réseau.
- X_entrée: Valeur de l'entrée correspondante.
- Si C = O, (C - O) est 0, donc les poids ne sont pas modifiés.
- Note: Modifier les poids peut altérer la classification correcte d'exemples précédemment bien classés, nécessitant plusieurs passes sur l'ensemble des exemples.

## Exercice Pratique : Apprentissage du "OU Booléen"
- Objectif: Entraîner un perceptron à apprendre la fonction logique OU.
- Entrées: x0=1 (biais), x1 et x2 (binaires: 0 ou 1).
- Sortie attendue: x1 OU x2.
- Processus: 1. Initialiser les poids arbitrairement (ex: 0, -1, 1 pour W0, W1, W2). 2. Parcourir séquentiellement les exemples du tableau d'entraînement. 3. Pour chaque exemple, calculer la sortie et mettre à jour les poids si nécessaire. 4. Repasser sur l'ensemble des exemples si des poids ont été modifiés, jusqu'à ce que les poids soient stables et que tous les exemples soient bien classés.
- Résultat stable (ex): Poids finaux de 0, 1, 1 pour W0, W1, W2 (pour le ou booléen).

## Deep Learning (Apprentissage Profond)
- Définition: Terme "à la mode" désignant des réseaux de neurones profonds, c'est-à-dire des réseaux avec plusieurs couches de neurones.
- Caractéristiques: Contiennent de nombreuses couches et un très grand nombre de paramètres (parfois des milliards dans les réseaux modernes).
- Principe fondamental: Le même que celui du perceptron simple (unités de calcul qui prennent, calculent et transmettent de l'information), mais avec une complexité accrue due à la profondeur.
- Succès: Capacité à entraîner des réseaux multicouches pour apprendre des décisions non linéaires complexes.
- Exemples d'architectures notables: LeNet-5 (1998): Conçu pour la reconnaissance de chiffres manuscrits (dataset MNIST). Contenait environ 60 000 paramètres. Structure typique: couches de convolution, couches de sous-échantillonnage, couche entièrement connectée, distribution de probabilité en sortie. AlexNet (2012): A révolutionné la classification d'images sur le benchmark ImageNet. Contenait environ 60 millions de paramètres. Inception (2014): Proposé pour ImageNet. Particularité: Utilise des convolutions de tailles différentes au même niveau de couche pour capturer l'information à diverses échelles. VGG16 (2015): Contenait environ 138 millions de paramètres.
- Évolution et facteurs clés: Augmentation constante de la profondeur des réseaux et du nombre de paramètres. Performance accrue du matériel (GPU): plus de puissance de calcul et de mémoire, permettant d'entraîner des modèles de plus en plus complexes et profonds.
- Architectures plus récentes: Transformers (2017): "Attention Is All You Need" (papier de recherche). Une architecture particulière qui a marqué une évolution importante.
- Le principe d'entraînement (ajustement des poids pour corriger les erreurs) reste le même, même si les architectures internes et les "briques" des réseaux deviennent plus complexes.

## Conclusion Générale
- Le concept d'entraînement des perceptrons, c'est-à-dire ajuster les poids pour corriger les erreurs, demeure la base fondamentale de tous les réseaux de neurones, y compris les plus profonds et complexes actuels.
- L'apprentissage profond applique ce principe pour entraîner des réseaux multicouches capables de résoudre des problèmes complexes et non linéaires.
- Ces techniques sont utilisées dans de nombreuses applications modernes (ex: architectures type Transformers).

# Time Series

## Introduction to Time Series
### What are Time Series?
- Definition: Ordered sequence of data points, typically measured at successive points in time.
- Key Property: Temporal order matters, unlike in tabular datasets.
- Examples: Air Passengers Time Series (1950-1960 data shown). Natural Sciences: Climate measurements, seismic activity. Engineering: Sensor readings, energy consumption data. Finance: Stock prices, exchange rates. Healthcare: Patient monitoring, ECG signals.

### Univariate and Multivariate Time Series
- Univariate Time Series: Observations of a single variable over time. Example: Daily temperature readings, stock prices.
- Multivariate Time Series: Observations of multiple variables over time. Captures relationships or dependencies between variables. Example: Weather data (temperature, humidity, wind speed), financial market (Stock 1, Stock 2, Stock 3 from 2023-01-01 to 2023-04-01).

### Comparison to Tabular Data: Importance of Ordering
- Tabular Data: No inherent order between rows.
- Time Series Data: Ordering of observations is crucial; it reflects temporal relationships. Patterns like trends, seasonality, and cyclic behaviors rely on temporal order. Reversing time order disrupts interpretability and prediction accuracy.

### Decomposition of Time Series
- Time Series Components: Trend: Long-term progression or direction in the data. Seasonality: Regular and repeating patterns over fixed time periods. Cyclic Patterns: Irregular, long-term fluctuations not tied to fixed intervals. Noise: Random variations or irregularities with no discernible pattern.
- Example: Sales data may exhibit an upward trend, annual seasonality, economic cycles, and random noise.

## Statistics-Based Feature Extraction
### Mean, Standard Deviation, and Z-Normalization
- Mean (µ): Average value of a time series. Formula: µ = (1/n) * Σ(xi)
- Standard Deviation (σ): Measure of the spread or dispersion of the data. Formula: σ = sqrt((1/n) * Σ(xi - µ)^2)
- Z-Normalization: Standardizes time series to have mean 0 and standard deviation 1. Formula: zi = (xi - µ) / σ. Removes scale and offset differences for better comparison.

### Temporal Auto-Correlation
- Definition: Measures the correlation of a time series with a lagged version of itself. Indicates how past values influence future values within the series.
- Formula: ρk = Σ(xt - µ)(xt+k - µ) / Σ(xt - µ)^2
- k: Lag; µ: Mean of the series.
- Applications: Identifying repeating patterns or dependencies; used in models like ARIMA and forecasting tasks.

### Fourier Transform for Time Series
- Definition: Decomposes a time series into its frequency components. Represents the series as a sum of sinusoidal functions (sines and cosines).
- Key Formula: F(f) = Σ(xt * e^(-2πi * ft/N))
- F(f): Complex frequency component at index f; xt: Signal value at time t.
- Applications: Identifying dominant frequencies in the data; filtering noise or periodicity detection.
- Note: Fourier Transform might be considered more reliable for outliers, as a perturbed mean might cause a "shift to the right".

### catch22: Canonical Time-Series Characteristics
- Definition: CAnonical Time-series CHaracteristics with 22 features.
- A set of handcrafted statistical features designed for time series classification and analysis.
- Key Features: Includes measures of distribution, autocorrelation, entropy, and non-linear properties (e.g., Mean autocorrelation, fluctuation analysis, entropy metrics).
- Advantages: Computationally efficient and interpretable; provides a compact feature set for quick analysis and classification tasks.

## Time Series Analysis
### General Time Series Analysis Tasks
- Classification, Clustering, Extrinsic Regression, Prototyping / Generation, Forecasting, Anomaly Detection.

### Time Series Forecasting/Regression
- Definition: Predicting future values based on past data. A regression problem where temporal ordering plays a key role.
- Key Idea: Given a time series X = {x1, x2, ..., xt}, predict xt+1, xt+2, ....
- Applications: Financial market prediction, demand forecasting, climate modeling and weather prediction.
- Note: Regression is mentioned possibly for 1% of the series when imputation is not possible.

#### Moving Average Model (MA)
- Definition: Statistical model predicting the next value as a linear combination of past forecast errors. Captures short-term dependencies.
- Key Formula: xt = µ + Σ(θi * ϵt-i) + ϵt
- µ: Mean of the series; ϵt: White noise at time t; θi: Parameters; q: Order of the model.
- Applications: Smoothing noisy time series data; short-term prediction in finance and weather.

#### Auto Regressive Model (AR)
- Definition: Model predicting the current value as a linear combination of its past values. Captures influence of previous time steps.
- Key Formula: xt = c + Σ(φi * xt-i) + ϵt
- c: Constant term; φi: Parameters; ϵt: White noise at time t; p: Order of the model.
- Applications: Modeling and forecasting time series with strong temporal dependencies (economics, finance, weather prediction).

### Time Series Imputation
- Definition: Filling in missing values within a time series.

#### Methods for Replacing Missing Values
- Replacing with Mean: Method: Compute mean µ of observed values, replace each missing value with µ. Advantages: Simple, computationally efficient, preserves overall average. Limitations: May not capture temporal structure, can distort patterns if missing values are frequent.
- Replacing with Median: Compute the median of observed values. Less sensitive to outliers compared to the mean; suitable for skewed distributions.
- Forward Fill (ffill): Replace missing values with the last observed value.
- Backward Fill (bfill): Replace missing values with the next observed value.
- Advantages (ffill/bfill): Preserves local trends and continuity. Useful in real-time data streams and sensor readings.

### Time Series Classification
- Definition: Assigning a category or label to a time series based on its patterns or features.

#### K-Nearest Neighbors (KNN) with Euclidean Distance (ED)
- Compares the distance between time series to find the closest match.
- Limitations: Requires series of equal length; insensitive to temporal shifting or frequency variations.

#### Dynamic Time Warping (DTW)
- Aligns two time series with shifting and scaling to measure similarity.
- Overcomes the limitations of KNN with ED by handling temporal distortions.
- Applications: Activity recognition, fault detection, medical diagnostics.
- Note: classification a valeur Continue refers to predicting a continuous label, covered under Extrinsic Regression.

### Handling Unequal Length Series
- Challenges: Time series often have different lengths due to varying sampling durations or missing data.
- Methods: Padding: Extend shorter series with zeros or other placeholder values. Truncation: Trim longer series to match the shortest series length. Dynamic Time Warping (DTW): Aligns time series of unequal length by allowing non-linear mappings of time indices, preserving important temporal patterns.

### Dimensionality Reduction/Symbolization of Time Series Data
#### Dimensionality Reduction
- Principal Component Analysis (PCA): Can reduce dimensions, but does not account for temporal ordering.
- Piecewise Aggregate Approximation (PAA): Reduces dimensionality by dividing the time series into equal-sized segments and averaging values within each segment.

#### Symbolization
- Symbolic Aggregate approXimation (SAX): Converts PAA segments into symbols for simplified representation.
- Symbolic Fourier Approximation (SFA): Utilizes Fourier Transform coefficients for symbolization.

## Conclusion
- Time series are temporally ordered datasets crucial in fields like healthcare, finance, and engineering, characterized by temporal dependency, seasonality, and trends.
- They can be Univariate or Multivariate.
- Key techniques include: Decomposition into components (trend, seasonality, cyclic patterns, noise). Statistical features (mean, standard deviation, Z-normalization, auto-correlation, Fourier Transform, catch22). Models like AR and MA for forecasting. Imputation methods for missing data (mean, median, forward/backward fill). Classification (KNN, DTW) and Extrinsic Regression (KNN-DTW, Random Forest, SVR, XGBoost). Methods for handling unequal length series (padding, truncation, DTW). Dimensionality reduction (PAA, PCA) and symbolization (SAX, SFA).

# Wooclap

## Intro (Data Mining Fundamentals)
- Primary objective of data mining: Discover patterns.
- Structured data example: A CSV file.
- Main purpose of data preprocessing: Clean data.
- Focus of exploratory data analysis (EDA): Understand data.
- Method for handling missing values by replacing them: Mean imputation.
- Technique for reducing data dimensionality: Principal Component Analysis (PCA).
- Language for managing relational databases: SQL.
- Type of unsupervised learning: Clustering.
- Purpose of big data analysis: Extract insights.
- Step involving assessing model quality: Evaluation.
- Key characteristic of unstructured data: Lacks a specific form.
- Discipline data mining intersects with for predictive analytics: Machine learning.
- "Veracity" in big data context: Data quality.
- Data visualization technique for showing variable distribution: Histogram.

## K-Nearest Neighbors (K-NN)
- Type of learning: Instance-based learning.
- Commonly used distance metric: Euclidean Distance.
- Primary disadvantage: High computational cost.
- Effect of small 'k': Sensitive to noise.
- Technique to ensure features contribute equally: Feature scaling.
- Type of voting assigning more influence to closer neighbors: Weighted voting.
- 'Lazy learning': No model building during training.
- Main purpose of one-hot encoding: Handle categorical features.
- Method for choosing optimal 'k': Cross-validation.
- Real-world application: Medical diagnosis.
- Key challenge with large datasets: High computational complexity.
- Scenario for weighted voting: When closer neighbors should have more influence.
- Critical preprocessing step for varying scales: Normalization.
- Strategy to avoid ties in binary classification: Choose an odd value for 'k'.

## Decision Trees
- Primary use: Classification and regression.
- Each internal node represents: A test on an attribute.
- Algorithm building trees using information gain: ID3 (Iterative Dichotomiser 3).
- 'Entropy': A measure of uncertainty.
- Overfitting: Learning the training data too well, including noise.
- Pruning process aim: Improve generalization by reducing complexity.
- Strategy to avoid overfitting by penalizing complexity: Cost Complexity Pruning (CCP).
- Common criterion to split nodes: Information gain.
- Random Forest's role in reducing overfitting: By averaging predictions of multiple trees.
- Data types handled effectively: Both numerical and categorical data.
- 'Minimum Description Length' principle in pruning: Prefer the simplest model that fits the data.
- 'Entropy gain': Reduction in uncertainty.
- Method for handling numerical data: Discretization.
- Problem from unrestricted tree growth: Overfitting.
- Why interpretable: Easy to understand and visualize.

## Bayes Classifier
- Bayes' Theorem used for: To calculate conditional probabilities.
- Naive Bayesian Classifier assumption about attributes: Independence.
- P(A): The probability of an event A.
- P(A ∩ B) for independent events: P(A)×P(B).
- P(A | B) represents: The probability of A given B.
- hMAP in Bayesian classification: Maximum A Posteriori Hypothesis.
- Attribute in text classification: A word in the text.
- Problem if feature value never appears in training set: Zero probability.
- Data handled well: Small datasets.
- Scenario not well suited for: Complex models with feature interactions.
- P(Ck) in Bayesian classification: The prior probability of class Ck.
- Method for very small probability values: Logarithmic transformation.
- Gaussian Naive Bayes assumption: Gaussian (Normal) distribution of attributes.
- Disadvantage of Naive Bayesian Classifier: Assumes conditional independence of features.
- Example of text classification: Spam email filtering.

## Hierarchical Clustering
- Primary goal of clustering: Divide a set of objects into groups.
- Clustering task type: Unsupervised.
- Common distance measure for numerical values: Euclidean distance.
- Purpose of scaling attributes: To ensure each attribute contributes equally.
- Dendrogram represents: Hierarchical representation of successive merges.
- Linkage method sensitive to noise: Single linkage.
- Main advantage of Ward's linkage: Creates balanced clusters.
- Determining number of clusters: Cut the tree at a specific height.
- Silhouette Coefficient type of evaluation criterion: Internal.
- Adjusted Rand Index compares: Clustering results to true labels.
- Major disadvantage: Computational cost of the distance matrix.
- Initial cluster formation: Each object is its own cluster.
- Centroid linkage method known for: Good resistance to noise.
- Number of clusters during process: Decreases over time.
- Criterion based on inertia: Ward's linkage.

## K-Means
- Primary objective: Minimize intra-cluster variance.
- First step: Choose k random points as cluster centers.
- Centroid update method: By calculating the mean of the points in each cluster.
- K represents: Number of clusters.
- Technique for improved centroid initialization: K-Means++.
- Convergence: When centroids no longer change.
- Common distance metric: Euclidean distance.
- Issue due to outliers: It distorts centroids.
- Method to decide 'k': Elbow Method.
- Common limitation: Assumes spherical clusters.
- Variant of K-Means: K-Medoids.
- Type of learning: Unsupervised learning.
- Silhouette Score's role: By evaluating the quality of clustering.
- Effect if 'k' is too large: Clusters may overlap too much.
- Main reason to use K-Means++: To improve initial centroid selection.

## Neural Networks
- Perceptron developed in 1960s by: Rosenblatt.
- Main limitation of perceptrons: Cannot solve non-linear problems.
- Backpropagation algorithm proposed in 1974 by: Werbos.
- Initial activation function in perceptrons: Heaviside.
- Key feature of LeNet-5: Designed for grayscale images.
- Distinguishes AlexNet from LeNet-5: Handles RGB images.
- Model introducing parallel filter sizes: Inception.
- VGG-16 learnable parameters: 138 million.
- Foundational principle to neural network training: Error correction.
- Inception technique to reduce computational cost: 1×1 convolutions.
- Common activation function in AlexNet: ReLU.
- Major achievement of AlexNet in 2012: Popularized deep learning.
- Primary advantage of deep neural networks: Non-linear decision boundaries.
- Problem of non-linear decision boundaries highlighted by: Minsky and Papert.
- Output of a perceptron passed through: Activation function.

## Time Series
- Key property of time series data: Temporal order matters.
- Z-Normalization: Standardizes the series to mean 0 and standard deviation 1.
- Primary goal of time series forecasting: Predict future values based on past data.
- Technique aligning time series of unequal lengths: Dynamic Time Warping (DTW).
- Component representing long-term progression: Trend.
- Imputation method using last observed value: Forward Fill.
- Purpose of Fourier Transform: Decomposes the series into frequency components.
- Method best for capturing cyclic patterns: Fourier Transform.
- Method filling missing values with median of observed values: Forward Fill.
- Key advantage of DTW in classification: It handles temporal distortions.
- Statistical feature capturing spread: Variance.
- Model predicting future value based on weighted sum of past values: Auto Regressive (AR) model."""
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
