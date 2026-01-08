# Auto-generated content module for Decision Tree
# Contains embedded markdown content for this topic

TOPIC_NAME = "Decision Tree"
TOPIC_KEY = "decision-tree"

CONTENT = {
    "decision-tree/1.md": """# Notes sur les Arbres de Décision

Les arbres de décision sont une famille d'algorithmes de fouille de données très utilisés et interprétables, permettant la classification (prédiction de valeurs discrètes) et la régression (prédiction de valeurs continues). Leur nature visuelle et la facilité d'explication de leurs décisions en font un outil précieux, notamment pour la communication avec des experts non-informaticiens.

## 1. Qu'est-ce qu'un Arbre de Décision ?

- **Définition**: Une structure d'arbre où chaque composant a une fonction spécifique.
- **Structure**:
  - **Nœuds**: Représentent des tests sur les attributs du jeu de données.
  - **Branches**: Correspondent aux valeurs possibles que peut prendre un attribut.
  - **Feuilles**: Représentent une classe (pour la classification) ou une valeur continue (pour la régression).

## 2. Jeu de Données Exemple: Météo et Jeu

- **Contexte**: Prédiction de la participation à un jeu (ex: golf/tennis) en fonction des conditions météorologiques.
- **Attributs (exemples)**:
  - État du ciel (Sky)
  - Température (Temperature)
  - Humidité (Humidity)
  - Vent (Wind)
- **Classe à prédire**: Jouer (Yes/No).
- **Processus d'inférence**: Parcourir l'arbre en fonction des conditions météo pour arriver à une feuille (décision Yes/No).

## 3. Construction de l'Arbre de Décision: Algorithme ID3

- **Algorithme principal**: ID3 (Iterative Dichotomiser 3)
- **Proposé par**: Quinlan en 1986.
- **Approche**: top-down (descendante), itérative.
- **Concepts clés**:
  - **Entropie**: Mesure l'incertitude d'un ensemble de données.
    - Maximale quand il y a équiprobabilité (ex: 0.5 probabilité pour deux événements, entropie = 1).
    - Nulle quand il n'y a aucune incertitude (ex: 100% de "Yes").
  - **Gain d'information**: Mesure la réduction de l'incertitude sur la classification suite à la connaissance d'un attribut.
    - L'objectif est de trouver l'attribut qui offre le plus grand gain d'information à chaque étape.
- **Processus de construction (simplifié)**:
  1. Calculer l'entropie de la classe de la population totale (entropie initiale).
  2. Pour chaque attribut, calculer le gain d'information qu'il apporte.
  3. Choisir l'attribut avec le gain d'information le plus élevé comme nœud racine (ou nœud interne).
  4. Répéter le processus récursivement pour chaque branche, jusqu'à ce que les feuilles soient pures (toutes de la même classe) ou qu'un critère d'arrêt soit atteint.
- **Exemple de calcul initial** (9 Yes, 5 No): Entropie de la population ≈ 0.94.

## 4. Gestion des Valeurs Numériques

- **Problème**: Impossible de créer une branche pour chaque valeur continue possible (ex: température en degrés Fahrenheit, humidité en pourcentage).
- **Solution**: Discrétisation
  - Transformer une variable continue en variable discrète (ex: température < 70 = "froid", 70-75 = "doux", > 75 = "chaud").
  - Peut être effectuée en pré-traitement des données.
  - Certains algorithmes (ex: C4.5) intègrent la discrétisation et choisissent les seuils de coupe optimaux pendant la construction de l'arbre.

## 5. Surapprentissage (Overfitting)

- **Définition**: Le modèle adhère trop aux données d'entraînement, ce qui le rend incapable de bien généraliser sur de nouvelles données.
- **Causes spécifiques aux arbres de décision**:
  - Profondeur excessive de l'arbre.
  - Trop de petites coupes spécialisées sur les données d'entraînement.
  - Données d'entraînement insuffisantes.
  - Absence d'élagage (pruning).
- **Stratégies d'Élagage (Pruning) pour l'éviter**:
  - Supprimer les nœuds qui n'améliorent pas significativement les performances.
  - Ajouter des scores de pénalité pour les arbres trop complexes.
  - Utiliser la théorie MDL (Minimum Description Length) pour trouver la description minimale optimale.
  - Techniques d'élagage basées sur les erreurs.
  - Contrôler la profondeur maximale de l'arbre.
  - Pré-élagage (contraintes pendant la construction) vs. Post-élagage (modification de l'arbre après construction).

## 6. Forêts Aléatoires (Random Forests)

- **Concept**: Une collection d'arbres de décision (ensemble learning).
- **L'idée est de construire plusieurs arbres et de combiner leurs décisions/prédictions pour obtenir une prédiction finale.**
- **Avantages**:
  - Très performantes et largement utilisées.
  - Peuvent surpasser l'apprentissage profond sur certaines applications spécifiques.
- **Diversité des arbres**: Essentielle car la construction d'un arbre unique (ex: avec ID3) est déterministe.
- **Méthodes pour introduire la diversité**:
  - Utiliser différents algorithmes de construction.
  - Appliquer différentes méthodes d'élagage.
  - **Échantillonnage des données**:
    - Échantillonnage des instances (ne pas utiliser toutes les données pour chaque arbre).
    - Échantillonnage des attributs (ne pas utiliser tous les attributs pour chaque arbre).

## 7. Avantages et Inconvénients des Arbres de Décision

- **Avantages**:
  - Interprétables et explicables (très visuels).
  - Faible préparation des données (normalisation souvent intégrée).
  - Gestion des données numériques et catégorielles.
  - Capacité à représenter des frontières de classe non linéaires.
- **Désavantages**:
  - Surapprentissage fréquent (sans élagage).
  - Instabilité: Les petits changements dans les données peuvent entraîner des modifications significatives de l'arbre.
  - Complexité de mise à jour: Nécessite souvent une reconstruction complète de l'arbre lors de l'ajout de nouvelles données.
  - Biais: Peut survaloriser des attributs avec de nombreuses valeurs possibles.
  - Optimum local: L'algorithme glouton de construction ne garantit pas l'optimum global de l'arbre.

## 8. Conclusion

Les arbres de décision, bien que potentiellement instables et sujets au surapprentissage, restent un modèle très pertinent, surtout pour les petits volumes de données. Les Random Forests, leurs dérivés, sont des algorithmes extrêmement performants et très utilisés dans diverses applications.""",
    "decision-tree/2.md": """# Decision Trees: Concepts, Construction, and Overfitting

Decision Trees are intuitive, flowchart-like models used for classification and regression tasks, easily visualized and interpreted. They are built by recursively splitting data based on attributes to reduce uncertainty, but can be prone to overfitting.

## 1. Introduction and Motivation

- **Definition**: A flowchart-like tree structure where:
  - Each internal node denotes a test on an attribute.
  - Each branch represents an outcome of the test.
  - Each leaf node holds a class label or continuous value (decision).
- **Purpose**: Used for both classification and regression.
- **Structure**:
  - Built from data.
  - Nodes: attributes.
  - Branches: values.
  - Leaves: decisions (classes).
- **Classification Process**: A new instance is tested by its path from the root to a leaf node.

## 2. ID3 (Iterative Dichotomiser 3)

- **Developer**: Ross Quinlan in 1986.
- **Purpose**: Create a decision tree to classify instances.
- **Approach**:
  - Top-down, greedy search through possible branches (no backtracking).
  - Selects the attribute that is most informative (highest information gain) as the decision node, proceeding recursively.
- **Attribute Selection**: Based on entropy and information gain.
- **Limitations**:
  - Can overfit the data.
  - Biased towards attributes with many outcomes.
  - Does not handle numeric attributes or missing values directly.

## 3. Criteria for Splitting

### Entropy Heuristic

- **Measures**: The amount of information or uncertainty in a set of examples (E).
- **Formula**:
  $$
  H(E) = -\\sum (p_i \\cdot \\log_2(p_i))
  $$
  Where \\( p_i \\) is the proportion of examples of class \\( i \\) in set \\( E \\).
- **Interpretation**:
  - Maximal Uncertainty: If classes are equally distributed (e.g., 4 yes, 4 no out of 8), \\( H(E) = 1 \\).
  - Zero Uncertainty: If all examples belong to one class (e.g., 8 yes out of 8), \\( H(E) = 0 \\).
  - Higher entropy implies greater uncertainty; lower entropy implies less uncertainty.

### Entropy Gain (Information Gain)

- **Measures**: Reduction in entropy (uncertainty) caused by partitioning a set of examples according to an attribute.
- **Formula**:
  $$
  G(a, E) = H(E) - \\sum \\left( \\frac{|E_{a,v}|}{|E|} \\cdot H(E_{a,v}) \\right)
  $$
  Where:
  - \\( E \\) is the set of examples.
  - \\( a \\) is the attribute.
  - \\( V(a) \\) is the set of values for attribute \\( a \\).
  - \\( E_{a,v} \\) is the subset of \\( E \\) where attribute \\( a \\) has value \\( v \\).
- **Objective**: The ID3 algorithm selects the attribute with the highest information gain to split the node.
- **Example Calculations**:
  - Initial population: 9 Yes, 5 No.
  - \\( H(E) \\approx 0.94 \\)
  - **Attribute Sky**:
    - Sunny (2 Yes, 3 No): \\( H(\\text{sunny}) \\approx 0.97 \\)
    - Overcast (4 Yes, 0 No): \\( H(\\text{overcast}) = 0 \\)
    - Rain (3 Yes, 2 No): \\( H(\\text{rain}) \\approx 0.97 \\)
    - \\( G(\\text{sky}, E) \\approx 0.94 - \\left( \\frac{5}{14} \\cdot 0.97 + \\frac{4}{14} \\cdot 0 + \\frac{5}{14} \\cdot 0.97 \\right) \\approx 0.246 \\)
  - **Attribute Humidity**:
    - High (3 Yes, 4 No): \\( H(\\text{high}) \\approx 0.98 \\)
    - Normal (6 Yes, 1 No): \\( H(\\text{normal}) \\approx 0.59 \\)
    - \\( G(\\text{humidity}, E) \\approx 0.94 - \\left( \\frac{7}{14} \\cdot 0.98 + \\frac{7}{14} \\cdot 0.59 \\right) \\approx 0.151 \\)
  - **Attribute Wind**:
    - Strong (3 Yes, 3 No): \\( H(\\text{strong}) = 1 \\)
    - Weak (6 Yes, 2 No): \\( H(\\text{weak}) \\approx 0.81 \\)
    - \\( G(\\text{wind}, E) \\approx 0.94 - \\left( \\frac{6}{14} \\cdot 1 + \\frac{8}{14} \\cdot 0.81 \\right) \\approx 0.048 \\)
  - **Conclusion**: Sky has the highest gain (0.246), so it would be chosen as the root node.

## 4. Handling Numerical Data

- **Challenge**: Cannot calculate gain for each continuous value directly.
- **Solution**: Discretize the variable (e.g., < 70 = Cold, [70-75] = Mild, > 75 = Hot).
- **Thresholds**: Can be found before or during tree construction (e.g., C4.5 algorithm).

## 5. Overfitting in Decision Trees

- **Definition**: Occurs when a model learns the training data too well, including noise and outliers, leading to poor generalization on new data.
- **Causes**:
  - Depth: Trees allowed to grow too deep create overly specific rules.
  - Complexity: Small splits deep in the tree capture noise.
  - Insufficient Data: Limited data leads to patterns that don't generalize.
  - No Pruning: Lack of pruning mechanisms.

### Pruning Strategies to Avoid Overfitting

- **Pruning**: Reducing tree size by converting branch nodes into leaf nodes, improving generalization.
- **Main Strategies**:
  - **Reduced Error Pruning**: Remove a node if it doesn't decrease prediction accuracy on a validation set.
  - **Cost Complexity Pruning (CCP)**: Introduces a penalty for complexity (number of leaf nodes) to minimize misclassification rate + penalty.
  - **Minimum Description Length (MDL)**: Prune based on Occam's razor (simplest model that fits the data).
  - **Minimum Error Pruning**: Prune nodes whose removal would not significantly increase the error rate.

## 6. Random Forest

- **Concept**: Builds multiple decision trees using subsets of the data.
- **Mechanism**: Averages the predictions of individual trees for the final decision.
- **Benefits**: Helps to reduce overfitting and can assess attribute importance.

## 7. Advantages and Disadvantages of Decision Trees

### Advantages

- **Interpretable**: Easy to understand and visualize.
- **Minimal Data Prep**: No need for normalization or scaling.
- **Handle Multiple Types**: Can deal with numerical and categorical data.
- **Non-linear Relationships**: Naturally handles non-linearity.

### Disadvantages

- **Overfitting**: Can become too complex, capturing noise.
- **Instability**: Small data changes can lead to different tree structures.
- **Bias**: Biased towards features with more levels.
- **Locally Optimal**: Greedy algorithms might not find the globally optimal tree."""
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
