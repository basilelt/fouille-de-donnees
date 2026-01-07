# Auto-generated content module for Clustering
# Contains embedded markdown content for this topic

TOPIC_NAME = "Clustering"
TOPIC_KEY = "clustering"

CONTENT = {
    "clustering/1.md": """# Notes sur le Clustering Hiérarchique

Le clustering hiérarchique est une tâche d'apprentissage non supervisé qui vise à construire des groupes (clusters) d'objets similaires, représentés par une structure arborescente (dendrogramme). Il se base sur des mesures de dissimilarité entre objets et la manière de calculer la distance entre les clusters, ce qui impacte fortement les résultats.

## 1. Introduction au Clustering

- **Définition**: Le clustering est une tâche non supervisée, contrairement à la classification qui est supervisée et utilise des données labellisées.
- **Objectif**: Construire des groupes d'objets (clusters) à partir de leur description, sans labels préexistants.
- **Principes**:
  - Maximiser la similarité intracluster (objets du même groupe doivent être les plus proches possible).
  - Minimiser la similarité intercluster (objets de groupes différents doivent être les plus éloignés possible).

## 2. Mesure de Dissimilarité (Distance)

- **Concept**: Définir une mesure pour évaluer de manière graduelle le degré de ressemblance entre deux objets.
- **Propriétés**: Pour être appelée "distance", cette mesure doit satisfaire certaines propriétés mathématiques.
- **Exemple (Clients)**: Marie, Bruno, Laurent décrits par l'âge et le salaire.
- **Visualisation dans un espace à deux dimensions**.
- **Distance Euclidienne**:
  - Souvent utilisée pour des vecteurs de valeurs numériques.
  - Calculée comme la racine carrée de la somme des différences au carré des caractéristiques.
- **Normalisation des Caractéristiques**:
  - Nécessaire avant le calcul de la distance (en particulier Euclidienne) pour éviter qu'une caractéristique avec une plus grande amplitude ne domine le calcul.
  - Objectif: Assurer que chaque caractéristique ait le même poids dans le calcul de la distance.
  - **Méthode: Normalisation Min-Max**:
    - Formule: $(valeur - min) / (max - min)$
    - Ramène toutes les caractéristiques dans le même domaine de définition (généralement entre 0 et 1).
- **Matrice de Distance**:
  - Matrice 2x2 qui stocke la distance entre chaque couple d'objets.
  - Complexité quadratique par rapport au nombre d'objets.
- **Choix de la Distance**: Dépend du type de données (ex: numérique, images, vidéos, sons, documents, utilisateurs). Il est crucial de définir une distance pertinente pour les objets à regrouper.

## 3. Algorithme de Clustering Hiérarchique Ascendant (Agglomératif)

- **Processus**:
  1. **Initialisation**: Chaque objet est un cluster unique (ex: 100 objets = 100 clusters).
  2. **Itération**:
     - Calculer la matrice de distance 2 à 2 entre tous les clusters.
     - Sélectionner les deux clusters les plus proches (distance minimale).
     - Fusionner ces deux clusters en un seul.
  3. **Terminaison**: Répéter jusqu'à ce que tous les objets soient dans un seul groupe (un seul cluster).
- **Dendrogramme**:
  - Représentation hiérarchique des fusions successives.
  - La hauteur des liens est proportionnelle à la distance à laquelle les clusters ont été fusionnés.
  - **Découpe**: Permet d'obtenir une partition en un nombre désiré de clusters en "coupant" le dendrogramme à une certaine hauteur. La forme du dendrogramme peut donner une indication sur le nombre de clusters.
- **Critère de Lien (Linkage) - Mesure de la distance entre groupes**: C'est un paramètre clé qui définit comment la distance entre deux clusters est calculée.
  - **Distance Minimum (Single Linkage)**: Distance entre les deux objets les plus proches de chaque cluster. Tendance à l'effet de "chaînage".
  - **Distance Maximum (Complete Linkage)**: Distance entre les deux objets les plus éloignés de chaque cluster.
  - **Distance Moyenne (Average Linkage)**: Moyenne de toutes les distances possibles entre paires d'objets des deux clusters. Coûteux.
  - **Centroïde**: Distance entre les centroïdes (moyennes) des clusters.
  - **Critère de Ward**: Basé sur l'inertie, tend à créer des clusters de cardinalités (nombre d'objets) similaires.
- Le choix du critère de lien entraîne des résultats de clustering différents même avec les mêmes données.

## 4. Exemples d'Utilisation

- **Contrainte**: La complexité quadratique de la matrice de distance limite son application aux grands volumes de données. Utilisé quand le nombre d'objets est "acceptable".
- **Applications**:
  - Bioinformatique: Regroupement de données d'expression de gènes.
  - Neurosciences: Clustering de représentations d'images issues d'IRM cérébrales pour étudier la représentation abstraite et visuelle.
  - Analyse comportementale: Clustering de séquences d'activités de chirurgiens (seniors vs. juniors) pour identifier des schémas opératoires.

## 5. Évaluation du Clustering

- **Différence avec la classification**: Pas toujours de vérité terrain (labels de classe) pour vérifier la qualité des clusters.
- **Deux approches principales**:
  - **Critères Internes**: Basés uniquement sur la qualité intrinsèque des clusters.
    - Mesurent la compacité (objets proches au sein d'un cluster) et la séparation (clusters distincts les uns des autres) dans l'espace des données.
    - Exemples: Coefficient de Silhouette, Dunn Index.
  - **Critères Externes**: Comparent la partition obtenue par l'algorithme à un clustering ou à des classes préexistantes (si disponibles).
    - Gèrent l'absence de correspondance directe entre les numéros de cluster (ex: Cluster 1) et les noms de classes (ex: Iris setosa), ainsi que la possibilité d'un nombre de clusters différent.
    - Exemples: Rand Index, Adjusted Rand Index.

## 6. Conclusion : Avantages et Inconvénients du Clustering Hiérarchique Ascendant

- **Avantages**:
  - **Simplicité et Intuitivité**: Facile à comprendre (processus itératif de regroupement).
  - **Flexibilité du nombre de classes**: Le nombre de clusters n'est pas un paramètre initial; il est choisi après la construction du dendrogramme par une simple coupe.
  - **Visualisation intuitive**: Le dendrogramme aide à orienter le choix du nombre de clusters.
- **Inconvénients**:
  - **Coût de la matrice de distance**: La complexité quadratique $O(n^2)$ (où n est le nombre d'objets) rend l'algorithme inadapté aux très grands volumes de données.
  - **Décisions irréversibles**: Les fusions de clusters ne sont jamais remises en question au cours de l'algorithme (approche gloutonne).
  - **Variabilité**: Les résultats dépendent fortement du critère de lien choisi (single, complete, average, Ward, etc.), ce qui peut rendre difficile le choix du "meilleur" critère.""",
    "clustering/2.md": """# Data Mining - Hierarchical Clustering

**Presented by Germain Forestier, PhD, Université de Haute-Alsace**

## Summary/Main Takeaway

Hierarchical clustering is an unsupervised data mining technique that organizes a set of objects into a hierarchy of clusters, often visualized as a dendrogram. It involves iteratively merging or splitting clusters based on defined dissimilarity measures and linkage criteria. While it provides intuitive visualization and flexibility in determining the number of clusters, it can be computationally intensive and results may vary depending on the chosen distance and linkage methods.

## 1. Introduction to Clustering

### Objectives:
- Divide a set of objects into groups (clusters).
- Maximize intra-cluster similarity and minimize inter-cluster similarity.
- Define a measure of dissimilarity (or distance) between objects.
- Clustering is an unsupervised task.

### Dissimilarity between Objects:
- Allows gradual evaluation of resemblance.
- Must satisfy mathematical properties to be called a distance.

#### For numerical values:
- **Euclidean distance:** $\\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$
- Requires scaling of attributes (e.g., Age vs. Salary).

**Scaling techniques:**
- **MinMax scaling:** $z = \\frac{x - \\min(x)}{\\max(x) - \\min(x)}$
- **Example:** Comparing clients (Marie, Bruno, Laurent) using scaled Age and Salary. $d(\\text{Marie}, \\text{Bruno}) \\approx 1.22$.

- Other distances exist for numerical values.

#### For other types of data:
- Specific distances for each data type.
- Objects can be described by multiple data types (e.g., image, video, sound, document, user).

## 2. Hierarchical Clustering Method

### Method Steps:
1. Start with one cluster per object.
2. At each step, two clusters are merged.
3. Cut the tree to obtain a partition.
4. Calculate the pairwise distance matrix between objects.
5. Select the smallest value in this matrix to identify the most similar groups.
6. Merge these two groups, replacing their rows with a row for the newly created group (based on a chosen criterion).
7. Repeat until a single group containing all objects is obtained.
8. This creates a sequence of partitions where the number $k$ of groups varies from $m$ (number of objects) to 1.
9. Finally, choose the value of $k$ using various possible methods.

### Dendrogram:
- A hierarchical representation of successive merges.
- The height of a link is proportional to the distance at which clusters were merged.

### Defining the Distance Between Clusters (Linkage Criteria):
1. **Minimum jump (single linkage):** Sensitive to noise.
2. **Maximum jump (complete linkage):** Tends to produce specific clusters; sensitive to noisy individuals.
3. **Average jump (average linkage):** Tends to produce clusters with similar variance.
4. **Centroid:** Good resistance to noise; not always possible to calculate.
5. **Ward:** Aggregation criterion based on inertia; creates "balanced" clusters.

**Note:** Different linkage criteria can lead to significantly different clustering results.

## 3. Example of Usage
- Clustering of gene expression data (Eisen et al., 1998).
- Clustering of similar image representation (Mur et al., 2013).
- Study of surgeon behavior (Forestier et al., 2012).

## 4. Clustering Evaluation

### Internal Evaluation:
- Evaluates the quality of clusters based solely on the data itself, without external labels.
- Focuses on compactness and separation of clusters.

#### Examples of Criteria:
- **Silhouette Coefficient:** $s(i) = \\frac{b(i) - a(i)}{\\max\\{a(i), b(i)\\}}$
  - $a(i)$: average distance between point $i$ and all other points in the same cluster.
  - $b(i)$: minimum average distance between point $i$ and points in another cluster.

- **Dunn Index:** $D = \\frac{\\min_{1 \\leq i < j \\leq k} \\delta(C_i, C_j)}{\\max_{1 \\leq l \\leq k} \\Delta(C_l)}$
  - $\\delta(C_i, C_j)$: distance between clusters $C_i$ and $C_j$.
  - $\\Delta(C_l)$: diameter of cluster $C_l$.

### External Evaluation:
- Compares clustering results to a ground truth or external labels.
- Measures how well the clustering matches the true classification.

#### Examples of Criteria:
- **Rand Index (RI):** $RI = \\frac{TP + TN}{TP + FP + FN + TN}$
  - $TP$: True Positives; $TN$: True Negatives.
  - $FP$: False Positives; $FN$: False Negatives.

- **Adjusted Rand Index (ARI):** $ARI = \\frac{RI - E[RI]}{\\max(RI) - E[RI]}$
  - $E[RI]$: Expected Rand Index of random clustering.

## 5. Conclusion

### Advantages:
- Easy to understand.
- Allows easy variation in the number of clusters.
- Intuitive visualization (dendrogram).

### Disadvantages:
- Computational cost of the distance matrix.
- The clustering criterion depends on the groups already formed.
- Different results depending on the clustering criterion (linkage method)."""
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
