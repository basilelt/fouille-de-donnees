# Auto-generated content module for Kmeans
# Contains embedded markdown content for this topic

TOPIC_NAME = "Kmeans"
TOPIC_KEY = "kmeans"

CONTENT = {
    "kmeans/1.md": """# Notes sur l'Algorithme K-means

L'algorithme K-means est une méthode de clustering (classification non supervisée) dont l'objectif est de construire automatiquement des groupes (ou clusters) compacts au sein des données. C'est un algorithme de partitionnement qui divise le jeu de données en K groupes distincts. Il est relativement simple à comprendre et très efficace.

## 1. Principes Fondamentaux de K-means

- **Classification Non Supervisée**: L'algorithme ne nécessite pas de données labellisées.
- **Partitionnement**: Divise un jeu de données en K sous-ensembles.
- **Paramètre K**: Le "K" dans K-means représente le nombre de clusters que l'on souhaite obtenir. Il doit être fixé a priori, contrairement au clustering hiérarchique.
- **Objectif**: Créer des clusters les plus compacts possible.
- **Moyenne (Means)**: Le terme "means" fait référence au calcul des moyennes pour définir les représentants des clusters.

## 2. Étapes de l'Algorithme K-means

L'algorithme est itératif et vise à améliorer une partition des données.

1. **Sélectionner K**: Choisir le nombre K de clusters à trouver.
2. **Initialisation**: Choisir aléatoirement K points dans l'espace des données pour servir de centres initiaux (ou centroïdes) des clusters. Des travaux existent pour des initialisations plus intelligentes que l'aléatoire simple.
3. **Affectation des objets**: Balayer toutes les données et affecter chaque objet à son cluster dont le centre est le plus proche.
4. **Recalcul des centres**: Pour chaque cluster, recalculer son nouveau centre en calculant la moyenne des objets qui lui ont été affectés. Pour des vecteurs numériques, c'est simple; pour des données plus complexes (ex: images), cela peut être plus délicat.
5. **Répétition**: Répéter les étapes 3 et 4 un certain nombre de fois. L'algorithme s'arrête lorsque:
   - Les résultats sont stables (les centres ne bougent plus, ou très peu d'objets changent de cluster).
   - Un nombre maximal d'itérations est atteint (ex: 10 à 15 itérations).

## 3. Exemple Pratique 1: Données Clients (2D)

- **Contexte**: Analyse de données clients avec "nombre de visites" et "nombre d'achats" (deux dimensions).
- **Processus**:
  1. Initialisation de 3 centres aléatoires.
  2. Affectation des clients (points) aux centres les plus proches.
  3. Recalcul des centres, qui se déplacent.
  4. Nouvelle affectation des clients aux nouveaux centres.
  5. Répétition jusqu'à la stabilisation des centres et des affectations.
- **Interprétation Sémantique**: Après clustering, un expert peut donner du sens aux groupes:
  - Groupe 1: Peu de visites, peu d'achats.
  - Groupe 2: Beaucoup de visites, beaucoup d'achats.
  - Groupe 3 (potentiellement intéressant): Beaucoup de visites, peu d'achats. Ce groupe peut être ciblé pour des promotions afin d'augmenter les achats.

## 4. Exemple Pratique 2: Images Histopathologiques

- **Contexte**: Analyse d'images médicales (biopsies) pour assister au diagnostic.
- **Application**: Clustering des pixels dans l'espace RGB (rouge, vert, bleu) en 3 dimensions, sans considérer la position spatiale des pixels.
- **Objectif**: Regrouper les pixels ayant des couleurs similaires.
- **Résultats**: Affichage de l'image segmentée pour K=2, 3, ou 4 clusters (couleurs).
- **Analyse**: L'expert attribue une sémantique aux clusters de couleurs (ex: cellules violettes sont des lymphocytes, cellules roses des macrophages), permettant de dériver des métriques (nombre de cellules, surface occupée) pour l'aide au diagnostic.
- **Limitation**: Fonctionne bien sur des images très contrastées; moins efficace sur des images avec des nuances de couleurs subtiles.

## 5. Aspects Importants et Considérations

### 5.1 Choix du Paramètre K

- **Délicat**: Impacte directement la qualité des clusters.
- **Méthodes**:
  - Intuition: Basée sur la visualisation ou la connaissance du domaine.
  - Essais multiples: Lancer l'algorithme plusieurs fois avec différentes valeurs de K et étudier les résultats.
  - Critères internes: Méthodes comme le critère du coude (elbow method) ou le score de silhouette.
  - Validation croisée: Tester sur des échantillons de données.
  - Connaissance expert: Demander à un spécialiste du domaine (ex: médecin).

### 5.2 Choix de la Distance (Métrique)

- **Essentiel**: Détermine comment la similarité entre objets est calculée.
- **Classique**: La distance euclidienne est la plus utilisée pour les vecteurs numériques.
- **Autres**: Distance de Manhattan, distance Cosine, etc. Le choix dépend de la nature des données et des caractéristiques à prendre en compte.
- **Exploration**: Il est possible de réaliser plusieurs clusterings avec différentes métriques pour observer les variations.

### 5.3 Critères d'Arrêt des Itérations

- **Stabilité des centres**: L'algorithme s'arrête lorsque les centres des clusters ne bougent plus significativement.
- **Stabilité des affectations**: Plus aucun ou très peu d'objets ne changent de cluster.
- **Erreur résiduelle**: Observer la compacité des clusters et arrêter quand elle n'évolue plus beaucoup.
- **Nombre d'itérations fixe**: Définir un nombre maximum d'itérations (ex: 10, 15).

### 5.4 Sensibilité aux Conditions Initiales

- **Optimisation Locale**: K-means est un algorithme d'optimisation locale.
- **Variabilité des Résultats**: Une initialisation aléatoire des centres peut conduire à des résultats différents à chaque exécution.
- **Solutions**:
  - Lancer K-means plusieurs fois et choisir le meilleur résultat.
  - Utiliser des variantes d'initialisation comme K-means++.

### 5.5 Sensibilité aux Valeurs Aberrantes

- **Impact**: Les valeurs aberrantes peuvent fortement influencer le calcul des moyennes et déformer les clusters.
- **Solutions**:
  - Pré-traitement des données (data cleaning).
  - Mise à l'échelle des données.
  - Utiliser des variantes de K-means moins sensibles, comme K-médoïdes.

## 6. Variantes de K-means

Il existe des centaines de variantes de K-means pour adresser ses limitations ou adapter à des besoins spécifiques.

- **K-médoïdes (K-Medoids)**:
  - Au lieu de calculer une moyenne abstraite comme centre, il choisit un objet existant des données du cluster comme représentant (médoïde).
  - Moins sensible aux valeurs aberrantes car le médoïde est une donnée réelle.
- **Fuzzy K-means (C-Means Flou)**:
  - Permet à un objet d'appartenir à plusieurs clusters avec un certain degré d'appartenance (partitionnement "flou" ou "doux").
  - Contrairement à K-means classique qui fait un partitionnement "dur" (un objet = un cluster).
- **K-means++**:
  - Améliore la phase d'initialisation des centres.
  - Sélectionne les centres initiaux de manière plus intelligente pour mieux couvrir l'espace de données, ce qui améliore la stabilité et la qualité des clusters finaux.
- **Bisecting K-means**:
  - Découpe itérativement les clusters trop grands en deux jusqu'à atteindre le nombre K désiré.

## 7. Avantages et Inconvénients de K-means

### 7.1 Avantages

- **Efficacité**: Algorithme rapide, surtout sur de grands jeux de données.
- **Interprétabilité**: Le centre de chaque cluster (la moyenne) fournit un prototype facile à interpréter par un expert.
- **Simplicité**: Peu de paramètres à définir (K, métrique, nombre d'itérations).

### 7.2 Inconvénients

- **Choix de K**: La détermination du nombre optimal de clusters (K) peut être difficile.
- **Contrainte de Prototype**: Avec la distance euclidienne, K-means tend à former des clusters de forme sphérique. Il peut mal performer sur des clusters de formes arbitraires.
- **Sensibilité à l'Initialisation**: En raison de son aspect d'optimisation locale, des initialisations défavorables peuvent mener à des résultats sous-optimaux. Il est recommandé de lancer l'algorithme plusieurs fois ou d'utiliser K-means++ pour plus de robustesse.
- **Sensibilité aux Aberrations**: Les points extrêmes peuvent fausser les centres des clusters.

## 8. Conclusion

K-means est un algorithme de clustering très connu, puissant et largement utilisé. Malgré ses inconvénients, il reste une base solide pour l'analyse de données non supervisée, en particulier grâce à sa simplicité et son efficacité. De nombreuses variantes existent pour pallier ses limitations et l'adapter à des contextes spécifiques.""",
    "kmeans/2.md": """# K-Means Clustering: Comprehensive Notes

## Main Takeaway
K-Means clustering is a widely used unsupervised learning technique that partitions data into k distinct, non-overlapping groups (clusters) based on similarity. Its primary objective is to minimize intra-cluster variance, thereby enhancing homogeneity within clusters. The algorithm operates iteratively, assigning data points to the nearest cluster centroid and then recomputing these centroids, continuing until a stopping criterion is met. While valued for its simplicity and efficiency, especially with large datasets, K-Means requires the number of clusters (k) to be predefined and is notably sensitive to the initial selection of centroids and the presence of outliers.

## 1. Introduction to K-Means Clustering
- **Source:** Germain Forestier, PhD, Université de Haute-Alsace.
- **Clustering:**
  - A technique in unsupervised learning.
  - Groups data based on similarity.
- **Partitioning Clustering:**
  - Divides the dataset into distinct, non-overlapping groups.
- **K-Means Clustering:**
  - A widely-used method where k specifies the number of clusters.
  - Objective: Minimize intra-cluster variance to enhance homogeneity.
  - Significance: Known for its simplicity and efficiency, especially in large datasets.

## 2. Understanding the K-Means Algorithm

### Overview
- Iteratively constructs and refines partitions to form satisfactory clusters.
- Involves calculating the mean to determine cluster centers (centroids).

### Steps of the K-Means Algorithm
1. Select k clusters to form.
2. Initialize by choosing k random points as cluster centers.
3. Assign each point to the nearest cluster center.
4. Recompute the centroids for each cluster by calculating the mean of all points assigned to that cluster.
5. Repeat the assignment and update steps until the stopping criterion is met (e.g., no points change clusters, a fixed number of iterations like 10-15 iterations).

### Exercise Example: E-commerce Customer Study
- **Objective:** Apply K-Means to study customers based on 'Number of Visits' and 'Number of Purchases'.
- **Data:** 10 clients with 2 characteristics (x, y) representing visits and purchases.
- **Initial Centers for 3 Clusters:**
  - Cluster 1: (1.5, 3.0)
  - Cluster 2: (4.0, 0.5)
  - Cluster 3: (2.5, 5.0)
- **Example Cluster Means (after initial iteration):**
  - Cluster 1: (1.50, 1.50)
  - Cluster 2: (4.66, 2.33)
  - Cluster 3: (5.33, 5.66)

## 3. Application Examples
- **Histopathological Image Processing:**
  - Goal: Automatically identify different structures in images captured with an electron microscope.
  - The colors in the image correspond to various structures.
  - Clustering results shown for 2, 3, and 4 clusters.

## 4. Fundamentals of K-Means

### Choosing 'k' in K-Means
- **Importance:** Directly impacts clustering quality; balances sensitivity to noise/outliers.
- **Methods for Determining 'k':**
  - Elbow Method: Identify the 'elbow' point in the plot of squared distances.
  - Silhouette Score: Higher scores indicate better-defined clusters.
  - Cross-validation: Assess cluster stability and predictive strength.
  - Practical Tips: Use domain knowledge; test various k values and evaluate.

### Distance Metrics in K-Means
- **Role:** Define similarity between objects; influence cluster shape and size.
- **Common Distance Metrics:**
  - Euclidean Distance: Straight line distance in space.
  - Manhattan Distance: Sum of absolute differences.
  - Cosine Similarity: Cosine of the angle between vectors.
- **Choosing the Right Metric:** Select based on data nature and clustering goals; experiment.

### Convergence of K-Means
- **Definition:** Occurs when cluster centroids stabilize between iterations.
- **Indicators of Convergence:**
  - Stable Centroids: No significant change in centroid positions (e.g., "stabilité 3 centres").
  - Assignment Stability: Points consistently belong to the same clusters.
  - Minimal Error Reduction: Small changes in the objective function.
- **Ensuring Convergence:** Use k-means++ for effective initial centroid selection; set a convergence threshold for centroid adjustments.

## 5. Practical Challenges and Solutions

### Common Issues with K-Means
- **Sensitivity to Initial Conditions:**
  - Clusters heavily depend on initial centroid selection, leading to variability.
  - Mitigation: Use k-means++ for better initial centroids.
- **Sensitivity to Outliers:**
  - Outliers can distort centroid calculation, resulting in biased clusters.
  - Mitigation: Use robust methods or consider algorithms like k-medoids.

### Improving K-Means Performance
- **Methods for Choosing Initial Centroids:**
  - k-means++: Optimizes initial centroid selection, leading to more consistent and efficient clustering, and reduces iterations.
- **Techniques for Outlier Handling:**
  - Data Cleaning: Remove noise and outliers before clustering.
  - Robust Scaling: Use scaling techniques less sensitive to outliers.

### Variants of K-Means
- **k-medoids:** Chooses actual data points (medoids) as centers, offering more robustness to outliers.
- **Fuzzy k-Means:** Each point belongs to all clusters with different degrees of membership, suitable for overlapping clusters.
- **k-means++:** (As mentioned above) Enhances the initialization phase of K-Means to improve cluster quality and convergence speed.
- **Bisecting k-Means:** A divisive clustering algorithm that iteratively splits clusters to achieve the desired number.

## 6. Advantages and Disadvantages of K-Means

### Advantages
- **Efficiency:** Operates in linear time, making it very fast.
- **Interpretability:** Clusters are easy to understand, centered around clear prototypes.
- **Simplicity:** Minimal parameters required (just the number of clusters and iterations).

### Disadvantages
- **Fixed k:** Number of clusters must be specified in advance.
- **Prototypical Constraints:** Limited to finding spherical-shaped clusters around centroids.
- **Initialization Sensitivity:** Outcome heavily depends on the initial position of centroids."""
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
