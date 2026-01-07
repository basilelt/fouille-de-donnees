# Auto-generated content module for Knn
# Contains embedded markdown content for this topic

TOPIC_NAME = "Knn"
TOPIC_KEY = "knn"

CONTENT = {
    "knn/1.md": """# Le Plus Proche Voisin (k-NN)

## Main Takeaway
Le k-Plus Proche Voisin (k-NN) est un algorithme d'apprentissage automatique basé sur des instances (lazy learning) qui permet de faire de la classification et de la régression en se basant sur la classe majoritaire ou la moyenne des k objets les plus proches dans l'ensemble d'entraînement. Simple à comprendre et à mettre en œuvre, il est adaptable mais présente des défis majeurs en termes de coût computationnel pour de grands volumes de données et nécessite une préparation rigoureuse des données, notamment la normalisation des caractéristiques et le choix optimal du paramètre k.

## Introduction
- Algorithme basé sur des instances (instance-based learning ou lazy learning).
- Permet de faire de la classification (prédire valeur discrète) et de la régression (prédire valeur continue). L'accent est mis sur la classification.
- Fonctionnement : Classifie un nouvel objet en se basant sur la classe majoritaire au sein des k plus proches voisins.
- k est le nombre de voisins à considérer et est un paramètre de l'algorithme.
- 1-Plus Proche Voisin (1-NN) : Cas particulier où on cherche uniquement l'objet le plus proche.
- Le k-NN est une famille d'algorithmes avec de nombreuses versions selon le choix de la distance, de k, de la pondération, etc.

## Concepts Clés

### Mesure de Distance
- Nécessaire pour définir la notion de "proximité".
- Distance Euclidienne : Très classique et largement utilisée pour les vecteurs de valeurs numériques.
- Autres distances : Manhattan, Minkowski, similarité cosinus. Le choix dépend du type de données et des caractéristiques souhaitées.

### Apprentissage Paresseux (Lazy Learning)
- Dit "paresseux" car il n'y a pas d'étape de construction de modèle à l'avance.
- Le "modèle" est constitué par l'intégralité des données d'entraînement.
- Lors de l'inférence (classification d'un nouvel objet), il est comparé à tous les objets du jeu d'entraînement.

## Historique et Applications
- Développé dans les années 1950.
- Formalisé par Cover et Hart en 1967.
- Applications courantes :
  - Vision par ordinateur / Reconnaissance d'images.
  - Systèmes de recommandation.
  - Médecine / Diagnostic médical (facilement explicable).
  - Recherche d'information.
- Implémenté dans l'intégralité des librairies d'apprentissage automatique.

## Avantages
- Simplicité : Facile à comprendre et à expliquer (ex: "chercher le plus proche").
- Polyvalence : Utilisable pour la classification et la régression.
- Rapidité de mise en place : Pas de phase de "training" coûteuse en temps de construction de modèle.
- Non-paramétrique : Ne fait pas d'hypothèses sur la distribution des données, peut construire des frontières de classe complexes.
- Mise à jour facile : Les nouvelles données sont simplement ajoutées au jeu d'entraînement et prises en compte instantanément pour les classifications futures, sans reconstruire de modèle.
- Explicabilité : Facile d'expliquer une prédiction en montrant les voisins les plus proches (particulièrement utile en diagnostic médical).

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
- Petit k : Sensible au bruit, risque de surapprentissage (overfitting), frontières de classification complexes.
- Grand k : Plus résistant au bruit, risque de sous-apprentissage (underfitting), frontières de décision plus lisses.
- Méthodes pour choisir k : Essai-erreur, validation croisée (analyse du taux d'erreur).
- Pour les données binaires, choisir un k impair (ex: 3 ou 5) aide à éviter les égalités de vote.

### Nécessité de Normalisation des Caractéristiques
- Indispensable pour que toutes les caractéristiques aient le même poids dans le calcul de distance.

## Exemple Visuel (Données IRIS)
- Utilisation des données IRIS : 150 objets (50 Setosa, 50 Vericolor, 50 Virginica).
- Un jeu d'entraînement et un jeu de test sont définis.
- Pour chaque objet du jeu de test, on cherche le plus proche voisin (ici, en 1-NN) dans le jeu d'entraînement pour assigner sa classe.

## Implémentation Naïve et Optimisations
- Fonction de classification naïve (1-NN) :

```
Pour chaque objet_test à classer:
    Trouver dans l'ensemble d'entraînement (train) l'objet_train le plus proche.
    Assigner la classe de l'objet_train trouvé à l'objet_test.
```

- Optimisations : Techniques pour accélérer le processus de recherche des voisins (ex: calcul de lower bound, pruning, mise en cache de mesures de distance).

## Pondération des Voisins
- Version de base : Vote à la majorité simple (tous les voisins ont le même poids).
- Alternative (pondération par distance) :
  - Attribuer un poids proportionnel à la distance de chaque voisin. Les voisins plus proches ont un poids plus important.
  - Utile pour résoudre les égalités de vote et donner plus de pertinence aux objets très proches.
  - Exemple de formule : \\( \\frac{1}{\\text{distance}^2} \\) (ou d'autres formules).

## Normalisation des Données (Mise à l'Échelle)
- Principe fondamental : Systématiquement mettre à l'échelle ou normaliser les données avant tout calcul de distance.
- Objectif : Ramener les caractéristiques à une même échelle (ex: entre 0 et 1).
- Exemple : La normalisation Min-Max est courante :

\\[
\\text{Valeur_normalisée} = \\frac{\\text{Valeur_originale} - \\text{Min_caractéristique}}{\\text{Max_caractéristique} - \\text{Min_caractéristique}}
\\]

- Raison : Sans normalisation, les caractéristiques avec de plus grandes échelles (ex: prix vs. taille en mètres carrés) domineraient le calcul de distance, indépendamment de leur importance réelle.

## Données Catégorielles
- Problème : Les mesures de distance standards sont conçues pour des valeurs numériques.
- Solution : One-Hot Encoding (encodage binaire).
- Convertit une caractéristique catégorielle en un vecteur numérique binaire.
- Ex: Une caractéristique "Couleur" avec les valeurs "red", "green", "blue" serait encodée en :
  - "red" -> [1, 0, 0]
  - "green" -> [0, 1, 0]
  - "blue" -> [0, 0, 1]
- Inconvénient : Peut augmenter considérablement le nombre de colonnes si une caractéristique prend beaucoup de valeurs possibles, ce qui impacte la complexité.

## Résumé et Points Clés
- Algorithme adaptable pour classification, régression, recommandation.
- Puissant et simple à expliquer, mais demande un travail significatif sur les données.
- Le choix de la mesure de distance est crucial et dépend du type de données.
- La complexité temporelle est le défi majeur avec de grands datasets.
- Le k-NN est une famille d'algorithmes avec de nombreuses variantes.
- Importance du prétraitement des caractéristiques : mise à l'échelle (normalisation) et gestion des données manquantes.
- Le paramètre k est central et doit être réglé avec soin.
- Avantages notoires : Explicabilité (ex: diagnostic médical) et facilité de mise à jour des données sans reconstruction du modèle.""",
    "knn/2.md": """# k-Nearest Neighbors (k-NN)

k-Nearest Neighbors (k-NN) is an instance-based learning algorithm applicable for both classification and regression tasks. It operates as a lazy learning method, meaning no explicit model is built during training; computations occur at the time of prediction. Its effectiveness relies on calculating distances between data points, making preprocessing steps like feature scaling and appropriate handling of categorical features crucial. Challenges include computational cost for large datasets and sensitivity to the choice of 'K' and noisy data.

## Definition and Overview

- **Definition**: An instance-based learning algorithm for classification and regression.
- **How it Works**: Classifies new data points based on the majority class (or average for regression) of its K nearest neighbors in the training data.
- **Distance Metrics**: Uses metrics like Euclidean, Manhattan, and Minkowski to measure closeness.
- **Lazy Learning**: No model is built during training; all computations occur during prediction.
- **Use Cases**: Applied in image recognition, recommender systems, and pattern recognition.

## Historical Background

- **Origins**: Developed in the 1950s, rooted in pattern recognition theory.
- **Formalized**: By Thomas Cover and Peter Hart in 1967.
- **Development**: Enhanced with weighted voting and various distance metrics.
- **Modern Usage**: Applied in computer vision, recommender systems, healthcare, and available in many ML libraries.
- **Challenges and Evolution**: Faces computational complexity with large datasets; ongoing research focuses on optimization and parallelization.

## Advantages and Disadvantages

### Advantages
- **Simplicity**: Easy to understand and implement.
- **Versatility**: Can be used for both classification and regression.
- **No Model Training**: Lazy learning approach.
- **Non-Parametric**: Makes no assumptions about underlying data distribution.

### Disadvantages
- **Computational Cost**: Expensive in terms of time and memory, especially with large datasets.
- **Sensitive to Noisy Data**: Outliers can negatively impact performance.
- **Choice of 'K'**: Selecting an optimal value for 'K' can be challenging.
- **Scaling Required**: Requires feature scaling to prevent distance bias.

## Understanding the k-NN Algorithm

### Pseudo Code: 1-Nearest Neighbor (1-NN)

This is a simplified version for k=1.

```python
function 1NN(X_train, Y_train, x):
    closestDistance = infinity
    closestClass = null
    # Compute distances between x and all samples in X_train
    for i = 1 to length(X_train):
        dist = distance(x, X_train[i])
        if dist < closestDistance:
            closestDistance = dist
            closestClass = Y_train[i]
    return closestClass
```

- **Input**: Training data (X_train, Y_train), test sample x.
- **Output**: Predicted class for the test sample x.
- **Distance**: A distance function (e.g., Euclidean, Manhattan).

## Distance Metrics

- **Euclidean Distance**: Commonly used, computed as the square root of the sum of squared differences.  
  $d(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$
- **Manhattan Distance**: Sum of the absolute differences between corresponding elements.  
  $d(x, y) = \\sum_{i=1}^{n} |x_i - y_i|$
- **Minkowski Distance**: Generalization of Euclidean (p=2) and Manhattan (p=1) distances.  
  $d(x, y) = \\left( \\sum_{i=1}^{n} |x_i - y_i|^p \\right)^{1/p}$
- **Cosine Similarity**: Measures the cosine of the angle between two vectors, often used with text data.  
  $\\text{similarity}(x, y) = \\frac{x \\cdot y}{\\|x\\| \\cdot \\|y\\|}$

## Choosing the Value of 'k'

The parameter K represents the "number of neighbors to inspect".

### Small Values of 'k'
- Sensitive to noise.
- Can lead to overfitting.
- Often results in a complex decision boundary.

### Large Values of 'k'
- More resistant to noise.
- Can lead to underfitting.
- Tends to produce a smoother decision boundary.

### Common Strategies
- **Cross-validation**: To test different 'k' values on the training set (do not test on the final test set).
- **Choosing an odd value for binary classification** to avoid ties.
- **Analyzing the error rate** for various 'k' values.

## Weighted vs. Unweighted Voting

### Unweighted Voting
- Every neighbor has an equal vote.
- Simple majority rule determines the class.

### Weighted Voting
- Neighbors' votes are weighted by their distance.
- Closer neighbors have a greater influence.
- **Example**: $w_i = \\frac{1}{d(x, x_i)^2}$ (weight is inverse square of distance).
- **Comparison**: Weighted voting offers more nuanced predictions than unweighted voting.

## Preprocessing for k-NN

### Feature Scaling and Normalization
- **Feature Scaling**: Brings all features to the same scale, preventing features with larger scales from dominating distance computations.
- **Normalization (Min-Max Scaler)**: A special case of scaling that transforms the range of features to [0, 1].  
  $x_{\\text{norm}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}$
- **Importance in KNN**: Ensures all features contribute equally to distance computation and prevents undue influence of features with larger numeric ranges.
- **Outliers**: Can significantly affect min-max scaling, potentially distorting the [0,1] range for the majority of data.

### Handling Categorical Features
- **Issue**: k-NN relies on distance computations, which are not directly applicable to categorical data.
- **Methods**:
  - **One-Hot Encoding**: Converts categories into binary (0 or 1) columns. This is suitable for nominal (unordered) categorical features.
  - **Ordinal Encoding**: Assigns ordered numbers to categories that inherently have an order (e.g., "un peu, beaucoup, à la folie").
- **Distance Metrics**: Hamming Distance is suitable for binary representations (e.g., after one-hot encoding).
- **Challenges**:
  - One-Hot Encoding can increase dimensionality ("sparse matrix").
  - Ensuring meaningful distances between categories.

## Conclusion
- **Versatile Algorithm**: Suitable for classification, regression, and recommendation systems.
- **Simple Yet Powerful**: Effective with appropriate feature engineering.
- **Distance Metrics**: Choice of metric is crucial for performance.
- **Challenges**: Computationally intensive (especially with large datasets), may struggle in high-dimensional spaces. However, it is adaptable and can be updated with new data easily, making it suitable for "big data" if optimized.
- **Advanced Techniques**: Includes Kernelized k-NN, ensembles (e.g., combining k=1, k=2, k=3, similar to Mixture of Experts in LLMs), etc.
- **Practical Considerations**: Feature scaling, handling missing data, and selecting an optimal 'k' are vital.
- **Real-world Applications**: Used in medical diagnosis, financial forecasting, and more."""
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
