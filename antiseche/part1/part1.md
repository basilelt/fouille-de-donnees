
# Synthèse Data Mining - Cours Germain Forestier

## 1. Introduction au Data Mining
Le Data Mining (fouille de données) consiste à découvrir des modèles, relations et connaissances dans de grands ensembles de données. Il combine statistiques, IA et Machine Learning.

### Processus CRISP-DM
1. **Business Understanding**: Définir les objectifs métier et critères de succès.
2. **Data Understanding**: Collecter, décrire et explorer les données (qualité).
3. **Data Preparation**: Nettoyage, fusion, transformation (80% du temps).
4. **Modeling**: Sélectionner et appliquer les algorithmes.
5. **Evaluation**: Vérifier si le modèle répond aux objectifs business.
6. **Deployment**: Mise en production et maintenance.

### Prétraitement et Qualité des Données
- **Données Structurées**: Tables (CSV, SQL). **Non structurées**: Texte, images, vidéos.
- **Nettoyage**: Gérer les valeurs manquantes (suppression ou imputation par moyenne/médiane/mode), détecter les outliers (valeurs aberrantes).
- **Transformation**: Normalisation Min-Max $[0, 1]$ ou Z-score (moyenne 0, écart-type 1) pour mettre les caractéristiques à la même échelle.
- **EDA (Exploratory Data Analysis)**: Visualisation (Histogrammes pour distribution, Boxplots pour dispersion/outliers, Nuages de points pour relations) pour comprendre les données avant modélisation.

### Concepts Statistiques et ML
- **Supervisé**: Données étiquetées (Classification pour valeurs discrètes, Régression pour valeurs continues).
- Ingénierie: Contrôle qualité, prédiction de pannes, optimisation des processus.
- Environnement et Agriculture: Prévisions météorologiques, rendements des cultures, gestion des ressources.

### Data Mining vs. Data Analysis vs. Machine Learning
- Data Analysis: Se concentre sur la compréhension des données et l'obtention d'informations via des méthodes statistiques.
- Data Mining: Va au-delà de l'analyse, utilisant des algorithmes pour trouver des modèles et prédire des tendances. Son périmètre inclut le prétraitement, l'analyse exploratoire et l'interprétation des résultats.
- Machine Learning (ML): Sous-ensemble de l'IA, entraîne des modèles à apprendre des données. Utilisé dans le Data Mining mais non synonyme.

### Intersections avec d'autres Disciplines
- Intelligence Artificielle (IA), Machine Learning, Big Data, Recherche d'Information, Reconnaissance de Formes.
- Text Mining, Gestion de Bases de Données, Analyse Prédictive, Traitement du Langage Naturel (NLP) (proche du TAL - Tout Analysis Language), Vision par Ordinateur (OCR).

### Relation AI, ML, Deep Learning
Le Deep Learning est un sous-ensemble du Machine Learning, qui est lui-même un sous-ensemble de l'Intelligence Artificielle.

### Domaine "Tendance"
Les termes comme "Machine Learning", "Big Data", "Data Science", "AI" sont souvent utilisés de manière interchangeable, parfois sur-médiatisés ou mal employés.

### Le "Data Scientist"
Profession très demandée nécessitant des compétences transdisciplinaires. (Aujourd'hui, il existe des profils comme AI engineer / AI scientist).

## 2. Types et Sources de Données

### Données Structurées vs. Non Structurées
- Données Structurées: Organisées dans un format ou un schéma défini (ex: tables, CSV). Facilement interrogeables.
  - Exemple: Jeu de données Iris (longueur/largeur sépale et pétale pour prédire l'espèce).
- Données Non Structurées: Manquent de forme spécifique (ex: documents texte, vidéos, images, posts réseaux sociaux). Nécessitent des techniques de traitement spécialisées.
- Problèmes: Classes mal balancées (ex: diagnostic de maladies rares où "pas malade" représente 99% de précision mais est inutile).

### Formats de Données Courants
- CSV (Comma-Separated Values): Texte brut, utilise des virgules pour séparer les valeurs.
- JSON (JavaScript Object Notation): Basé sur du texte, lisible par l'homme, pour échanger des données web.
- XML (eXtensible Markup Language): Langage de balisage, structure auto-descriptive, services web.
- Autres: YAML, HDF5.

### Sources de Données
- Bases de Données: Repositories structurés (relationnelles, documentaires, clé-valeur).
- Web Scraping: Extraction de données de sites web (peut être automatisée).
- APIs (Application Programming Interfaces): Protocole pour construire et interagir avec des applications logicielles, souvent pour récupérer des données web.

## 3. Prétraitement des Données

### Importance de Données Propres et de Qualité
- Données Propres: Sans erreurs ni incohérences, améliore la précision des modèles analytiques, facilite l'intégration.
- Données de Qualité: Respecte des normes (précision, complétude, fiabilité), soutient la prise de décision, impacte le succès des projets.
- Conséquences d'une mauvaise qualité: Conclusions incorrectes, augmentation des coûts opérationnels, diminution de la confiance.
- Notes de séance: La préparation des données prend beaucoup de temps. Les données ne sont pas toujours disponibles ou utilisables (nécessitent un cleaned dataset).

### Techniques de Nettoyage des Données
- Gestion des valeurs manquantes:
  - Supprimer les enregistrements.
  - Imputer les valeurs (moyenne, médiane, mode).
  - Utiliser des modèles ML pour prédire les valeurs manquantes.
- Transformation des données: Normalisation/mise à l'échelle, conversion de types/formats, encodage de variables catégorielles.
- Détection d'aberrations (Outliers): Identifier et gérer les valeurs anormales (méthodes statistiques ou ML). À faire avec prudence car elles peuvent contenir des informations importantes.
- Jeu d'entraînement et jeu de test: Les données aberrantes sont pertinentes ici.
- Validation et Vérification: S'assurer de la précision des données, vérifier avec des sources externes fiables.

## 4. Analyse Exploratoire des Données (EDA)

### But de l'EDA
- Comprendre la structure des données: Analyser la forme, la tendance centrale, la dispersion, visualiser les relations entre variables.
- Identifier les modèles et anomalies: Découvrir les tendances, clusters, outliers, erreurs potentielles.
- Faciliter la communication: Utiliser des visualisations pour rendre les données complexes compréhensibles.

### Techniques de Visualisation des Données
- Diagrammes à barres: Fréquence ou décompte par catégorie.
- Nuages de points: Relation entre deux variables continues.
- Graphiques linéaires: Tendances au fil du temps.
- Histogrammes: Distribution d'une variable continue.
- Boîtes à moustaches (Box Plots): Tendance centrale et variabilité.
- Considération: Choisir la technique appropriée en fonction du type de données et de la question.

## 5. Outils et Logiciels pour le Data Mining

### Outils Populaires
- Python: Langage généraliste, écosystème riche (ex: pandas, matplotlib, scikit-learn, NumPy).
- R: Langage pour le calcul statistique, vaste bibliothèque de packages.
- SQL: Langage de requête pour bases de données relationnelles (manipulation, agrégation, jointures).
- Excel: Logiciel de tableur, outils intégrés pour petites analyses.

### Bibliothèques et Packages
- Pandas: Manipulation et analyse de données, structures de données efficaces (dataframe).
- NumPy: Calcul numérique, grands tableaux et matrices.
- Matplotlib: Visualisation (statique, interactive, animée).
- Scikit-learn: Machine Learning (classification, régression, clustering, prétraitement).

## 6. Concepts de Base en Statistique et Machine Learning

### Statistiques Descriptives vs. Inférentielles
- Descriptives: Résument les aspects principaux d'un jeu de données (moyenne, médiane, mode, écart-type). Fournit un aperçu.
- Inférentielles: Fait des prédictions ou des inférences sur une population à partir d'un échantillon (tests d'hypothèses, intervalles de confiance, régression). Tire des conclusions générales.

### Apprentissage Supervisé vs. Non Supervisé
- Supervisé: Nécessite un jeu de données étiqueté (paires entrée-sortie). Le but est d'apprendre une fonction qui mappe les entrées aux sorties (ex: classification, régression). Utilise des données passées avec un label pour prédire le futur.
- Non Supervisé: Travaille avec des données non étiquetées. Le but est d'identifier des modèles, structures ou relations (ex: clustering, réduction de dimensionnalité, règles d'association). Utilisé quand il n'y a pas de label (ex: description). Algorithmes auto-supervisés (self-supervised) comme les LLMs créent leurs propres tâches (ex: masquer du texte dans une phrase).

### Compromis Biais-Variance (Bias-Variance Tradeoff)
- Biais: Erreur due à la simplification excessive du modèle. Un biais élevé entraîne un sous-apprentissage (underfitting), ne parant pas à capturer le modèle sous-jacent.
- Variance: Erreur due à la sensibilité aux petites fluctuations des données. Une variance élevée entraîne un sur-apprentissage (overfitting), capturant le bruit plutôt que le modèle.
- Compromis: Réduire le biais augmente la variance, et vice-versa. L'objectif est d'équilibrer les deux pour minimiser l'erreur totale. La précision n'est pas la seule métrique; l'interprétabilité est aussi cruciale.

## 7. Le Processus de Data Mining (CRISP-DM)

1. Compréhension de l'entreprise (Business Understanding): Définir les objectifs et les buts métier.
   - Importance: Assure l'alignement avec les stratégies, clarifie le problème, définit les métriques de performance, optimise l'allocation des ressources, facilite la communication, évalue l'impact, gère les risques.
2. Compréhension des données (Data Understanding): Collecter, décrire, explorer et vérifier la qualité des données.
3. Préparation des données (Data Preparation): Nettoyer, transformer, intégrer, sélectionner et formater les données.
4. Modélisation (Modeling): Sélectionner les techniques, concevoir les tests, construire et évaluer les modèles.
5. Évaluation (Evaluation): Évaluer la qualité du modèle, réviser le processus, déterminer les étapes suivantes.
6. Déploiement (Deployment): Planifier le déploiement, la maintenance et la surveillance du modèle.

## 8. Introduction au Big Data et à la Scalabilité

### Comprendre le Big Data
- Caractérisé par les "5 V":
  - Volume: Grandes tailles de données (téraoctets, pétaoctets).
  - Vélocité: Vitesse de génération et de traitement des données.
  - Variété: Types de données (structurées, semi-structurées, non structurées).
  - Véracité: Qualité et fiabilité des données.
  - Valeur: Valeur potentielle dérivée des données.
- Défis: Stockage, traitement, analyse, sécurité.
- Technologies: Hadoop, Spark, bases de données NoSQL.

### Défis liés au Big Data
- Stockage: Gestion des grands volumes, distribution, redondance.
- Traitement: Traitement efficace, souvent via le calcul parallèle.
- Intégration: Fusion de données de sources et formats divers.
- Qualité: Assurer la précision, la cohérence et la fiabilité.
- Sécurité: Protection de la vie privée, intégrité, conformité réglementaire.
- Analyse: Extraction d'insights à partir de jeux de données complexes et diversifiés.
- Scalabilité: Mise à l'échelle des systèmes pour gérer la croissance des données sans perte de performance.
- Coût: Gérer les coûts de stockage, de traitement et d'analyse par rapport à la valeur obtenue.
- Problématique: Les algorithmes sont durs à exécuter sur des ensembles de données énormes; on travaille souvent sur des sous-ensembles.

## 9. Exemples Concrets et Études de Cas

### Applications Réussies du Data Mining
- Santé: Prédiction d'épidémies, personnalisation des traitements.
- Finance: Détection de fraudes, gestion des risques.
- Commerce de détail: Recommandations de produits, optimisation des prix, gestion des stocks (ex: Walmart).
- Fabrication: Contrôle qualité, optimisation des processus.
- Transport: Prédiction du trafic, optimisation d'itinéraires (ex: Google Maps), maintenance prédictive (ex: GE Aviation).
- Énergie: Prévision de la demande.
- Divertissement: Recommandations de contenu (ex: Netflix, plus de 75% des vues).
- Gouvernement: Amélioration de la sécurité publique, prestations de services (ex: National Weather Service).

### Exemples Concrets de Succès
- Netflix: Algorithmes de recommandation personnalisée.
- American Express: Analyse des transactions pour la détection de fraude.
- Walmart: Optimisation des niveaux de stock.
- GE Aviation: Maintenance prédictive des moteurs d'avion.
- Google Maps: Analyse du trafic en temps réel.
- IBM Watson en Santé: Aide au diagnostic et à traitement du cancer.
- National Weather Service: Amélioration des prévisions météorologiques.
- LinkedIn: Suggestion de connexions professionnelles et d'opportunités d'emploi.

# Cours par Germain Forestier, PhD, Université de Haute-Alsace. QCM de dernière séance: 1 seule réponse, pas de points négatifs.

# Bayes

## Résumé
Le classifieur bayésien utilise les probabilités, notamment le théorème de Bayes et l'hypothèse d'indépendance des attributs, pour attribuer une classe à une instance donnée. Il vise à trouver l'hypothèse (classe) la plus probable en se basant sur les observations du jeu d'entraînement, ce qui le rend utile pour diverses tâches de classification, y compris l'analyse de texte.

## 1. Introduction au Classifieur Bayesien
- Objectif: Utiliser les probabilités pour la tâche de classification.
- Principe: Affecter à chaque hypothèse (classe) une probabilité d'être la bonne solution.
- Méthode: Observer des instances d'entraînement pour modifier les distributions de probabilité.
- Tâche: Trouver l'hypothèse la plus probable (la classe la plus probable) étant donnée une instance.
- Base théorique: S'appuie sur les probabilités conditionnelles et le théorème de Bayes.
- Particularité: Fait une hypothèse d'indépendance des attributs pour simplifier les calculs.

## 2. Rappels de Probabilités
- Probabilité d'un événement (PA): Entre 0 et 1.
- Événement certain: P = 1.
- Événement impossible: P = 0.
- Indépendance de A et B: P(A ∩ B) = P(A) * P(B)
- Probabilité de non-A: P(¬A) = 1 - P(A)
- Probabilité Conditionnelle (P(A sachant B)): Probabilité que A apparaisse, sachant que B est apparu.
- P(A | B) = P(A ∩ B) / P(B)
- D'où P(A ∩ B) = P(A | B) * P(B)
- Également P(A ∩ B) = P(B | A) * P(A)
- Événements Indépendants: Si A et B sont indépendants, P(A | B) = P(A). La connaissance de B ne modifie pas la probabilité de A.

## 3. Théorème de Bayes
- Formule: P(A | B) = [P(B | A) * P(A)] / P(B)

## 4. Application à la Classification (Fouille de Données)
- But: Calculer pour chaque classe C_k la probabilité qu'elle soit la solution sachant une instance X (description d'objet).
- Exemple: Pour un iris, P(setosa | pétal_length, sépal_length, ...)
- Formule appliquée: P(Classe | Description) = [P(Description | Classe) * P(Classe)] / P(Description)
- P(Classe | Description): Probabilité à postériori (ce que l'on cherche).
- P(Description | Classe): Probabilité de la description sachant la classe (vraisemblance).
- P(Classe): Probabilité a priori de la classe.
- P(Description): Probabilité de l'instance observée.
- Estimation des probabilités à partir du jeu d'apprentissage:
  - P(Classe): Proportion d'instances de cette classe dans le jeu d'entraînement.
  - Ex: Pour 50 Setosa sur 150 iris, P(Setosa) = 50/150 = 1/3.
  - P(Description): Proportion d'instances ayant ces valeurs d'attributs.
  - P(Description | Classe): Nombre de fois où cette description est constatée dans la classe C_k / Nombre total d'instances de la classe C_k.

### 4.1 Observations sur la Formule
- P(Classe | Description) augmente si P(Classe) augmente (classe plus fréquente).
- P(Classe | Description) augmente si P(Description | Classe) augmente (description fréquente dans cette classe).
- P(Classe | Description) diminue si P(Description) augmente (description est très courante toutes classes confondues, donc peu informative).

### 4.2 Maximum A Posteriori (MAP)
- On calcule P(Classe_k | Description) pour chaque classe C_k.
- On choisit la classe C_k avec la probabilité à postériori la plus élevée (arg max).
- Simplification: Le dénominateur P(Description) est une constante pour toutes les classes car il ne dépend pas de C_k.
- On peut donc le retirer pour trouver le maximum:
- argmax_k P(Classe_k | Description) = argmax_k [P(Description | Classe_k) * P(Classe_k)]

## 5. Hypothèse du Bayesien Naïf: Indépendance des Attributs
- Hypothèse: Les attributs sont indépendants les uns des autres sachant la classe.
- Conséquence: P(Description | Classe) = P(a_1, a_2, ..., a_n | Classe) peut être décomposé en un produit de probabilités:
- P(Description | Classe) = P(a_1 | Classe) * P(a_2 | Classe) * ... * P(a_n | Classe)
- Formule finale pour le classifieur Bayesien Naïf:
- argmax_k [P(Classe_k) * ∏_{i=1 à n} P(a_i | Classe_k)]
- Estimation pour attributs discrets:
- P(a_i | Classe_k) = (Nombre d'instances de Classe_k ayant la valeur a_i pour l'attribut i) / (Nombre total d'instances dans Classe_k).

## 6. Exemple 1: Prédiction de popularité d'un jeu
- Jeu d'entraînement: Jeux avec Genre (RPG/Action), Plateforme (Console/PC/Mobile), Budget Marketing (Faible/Moyen/Haut), et Populaire (Oui/Non).
- Nouvelle instance à classer (X): Genre=RPG, Plateforme=PC, Budget Marketing=Medium.
- Tâche: Prédire si le jeu sera Populaire=Oui ou Populaire=Non.

### 6.1 Calculs pour Populaire=Oui (Yes)
- P(Yes): Proportion de jeux "Oui" dans le jeu d'entraînement.
- Si 3 "Oui" sur 5 jeux: P(Yes) = 3/5.
- P(RPG | Yes): Proportion de jeux "Oui" qui sont RPG.
- Ex: 2/3.
- P(PC | Yes): Proportion de jeux "Oui" qui sont sur PC.
- Ex: 1/3.
- P(Medium | Yes): Proportion de jeux "Oui" avec budget Medium.
- Ex: 1/3.
- Calcul: P(Yes | X) = P(Yes) * P(RPG | Yes) * P(PC | Yes) * P(Medium | Yes)
- P(Yes | X) = (3/5) * (2/3) * (1/3) * (1/3) = 0.0444 (valeur non normalisée)

### 6.2 Calculs pour Populaire=Non (No)
- P(No): Proportion de jeux "Non".
- Si 2 "Non" sur 5 jeux: P(No) = 2/5.
- P(RPG | No): Proportion de jeux "Non" qui sont RPG.
- Ex: 1/2.
- P(PC | No): Proportion de jeux "Non" qui sont sur PC.
- Ex: 1/2.
- P(Medium | No): Proportion de jeux "Non" avec budget Medium.
- Ex: 1/2.
- Calcul: P(No | X) = P(No) * P(RPG | No) * P(PC | No) * P(Medium | No)
- P(No | X) = (2/5) * (1/2) * (1/2) * (1/2) = 0.05 (valeur non normalisée)

### 6.3 Prédiction
- P(No | X) (0.05) > P(Yes | X) (0.0444).
- Prédiction: Le jeu ne sera pas populaire.

### 6.4 Normalisation des Probabilités
- Les probabilités obtenues par produit sont souvent très petites.
- Méthode: Diviser chaque probabilité par la somme de toutes les probabilités des classes.
- Somme: 0.0444 + 0.05 = 0.0944
- P_norm(Yes | X) = 0.0444 / 0.0944 = 0.47 (47%)
- P_norm(No | X) = 0.05 / 0.0944 = 0.53 (53%)
- Avantage: Donne une distribution de probabilité sommée à 1, plus interprétable et donne une notion de certitude (ex: 53% vs 90%).

## 7. Gestion des Très Petites Probabilités
- Avec un grand nombre d'attributs (ex: 100), le produit de probabilités inférieures à 1 peut donner des valeurs extrêmement petites.
- Solution: Travailler dans un espace logarithmique.
- Transformation: log(a * b) = log(a) + log(b).
- Permet de transformer un produit en une somme, évitant les sous-dépassements numériques et les valeurs trop petites.

## 8. Gestion des Valeurs Numériques
- Problème: Impossible de calculer P(valeur_numérique | Classe) pour chaque valeur exacte (température, taille, etc.) car trop de valeurs possibles, ou trop peu d'instances pour chaque valeur.
- Solution: Estimer une distribution de probabilité pour les attributs numériques.
- Le plus simple: faire l'hypothèse d'une distribution Gaussienne (normale).
- On estime la moyenne (μ) et l'écart-type (σ) de l'attribut numérique pour chaque classe.
- La fonction de densité de probabilité gaussienne est ensuite utilisée pour estimer P(valeur_numérique | Classe).

## 9. Applications en Analyse de Texte
- Très utilisé pour la classification de texte (ex: détection de spam, attribution d'auteur, classification thématique).
- Défi: Représenter le texte non structuré en format attribut-valeur.
- Approche 1 (Naive): Chaque mot à chaque position est un attribut.
- Problème: Trop d'attributs et de valeurs de probabilité à estimer.
- Approche 2 (Bag-of-Words): S'abstraire de la position des mots.
- Représentation: Chaque mot du vocabulaire est un attribut. La valeur de l'attribut est soit sa présence/absence, soit son nombre d'occurrences (fréquence) dans le texte.
- Permet d'estimer les probabilités (P(mot | Classe)) en calculant les fréquences des mots dans les textes de chaque classe.

## 10. Avantages du Classifieur Bayesien Naïf
- Facile à implémenter.
- Relativement efficace.
- Interprétable: Produit des distributions de probabilité qui donnent une confiance dans la classification.
- Performant avec de petits jeux de données.
- Passe bien à l'échelle: Peut gérer un très grand nombre de caractéristiques.

## 11. Inconvénients et Limitations
- Hypothèse d'indépendance des attributs: Rarement vraie dans la réalité.
- Peut être modélisée, mais rend l'algorithme plus complexe et coûteux.
- Problème des Zéro Probabilités:
  - Si une valeur d'attribut n'apparaît jamais pour une classe, P(attribut | Classe) sera 0.
  - Étant donné que toutes les probabilités sont multipliées, cela annule le score de la classe, même si d'autres attributs suggèrent fortement cette classe.
  - Solution pratique: Utiliser un petit epsilon (ε) ou une technique de lissage (ex: lissage de Laplace) au lieu de 0 pour les probabilités nulles.
- Modèles complexes: Moins adapté aux modèles avec beaucoup d'interactions complexes entre les caractéristiques.
- Des extensions existent pour modéliser ces interactions, mais sont plus coûteuses.

Le Bayesien Naïf est une base solide et performante pour des problématiques simples, avec de nombreuses évolutions pour des cas plus complexes.

# Clustering

## Introduction au Clustering
- Définition: Tâche non supervisée, contrairement à la classification qui est supervisée et utilise des données labellisées.
- Objectif: Construire des groupes d'objets (clusters) à partir de leur description, sans labels préexistants.
- Principes: Maximiser la similarité intracluster (objets du même groupe doivent être les plus proches possible). Minimiser la similarité intercluster (objets de groupes différents doivent être les plus éloignés possible).

## Mesure de Dissimilarité (Distance)
- Concept: Définir une mesure pour évaluer de manière graduelle le degré de ressemblance entre deux objets.
- Propriétés: Pour être appelée "distance", cette mesure doit satisfaire certaines propriétés mathématiques.
- Exemple (Clients): Marie, Bruno, Laurent décrits par l'âge et le salaire.
- Visualisation dans un espace à deux dimensions.
- Distance Euclidienne: Souvent utilisée pour des vecteurs de valeurs numériques. Calculée comme la racine carrée de la somme des différences au carré des caractéristiques.
- Normalisation des Caractéristiques: Nécessaire avant le calcul de la distance (en particulier Euclidienne) pour éviter qu'une caractéristique avec une plus grande amplitude ne domine le calcul. Objectif: Assurer que chaque caractéristique ait le même poids dans le calcul de la distance.
- Méthode: Normalisation Min-Max: Formule: (valeur - min) / (max - min). Ramène toutes les caractéristiques dans le même domaine de définition (généralement entre 0 et 1).
- Matrice de Distance: Matrice 2x2 qui stocke la distance entre chaque couple d'objets. Complexité quadratique par rapport au nombre d'objets.
- Choix de la Distance: Dépend du type de données (ex: numérique, images, vidéos, sons, documents, utilisateurs). Il est crucial de définir une distance pertinente pour les objets à regrouper.

## Algorithme de Clustering Hiérarchique Ascendant (Agglomératif)
- Processus: 1. Initialisation: Chaque objet est un cluster unique (ex: 100 objets = 100 clusters). 2. Itération: Calculer la matrice de distance 2 à 2 entre tous les clusters. Sélectionner les deux clusters les plus proches (distance minimale). Fusionner ces deux clusters en un seul. 3. Terminaison: Répéter jusqu'à ce que tous les objets soient dans un seul groupe (un seul cluster).
- Dendrogramme: Représentation hiérarchique des fusions successives. La hauteur des liens est proportionnelle à la distance à laquelle les clusters ont été fusionnés. Découpe: Permet d'obtenir une partition en un nombre désiré de clusters en "coupant" le dendrogramme à une certaine hauteur. La forme du dendrogramme peut donner une indication sur le nombre de clusters.
- Critère de Lien (Linkage) - Mesure de la distance entre groupes: Distance Minimum (Single Linkage): Distance entre les deux objets les plus proches de chaque cluster. Tendance à l'effet de "chaînage". Distance Maximum (Complete Linkage): Distance entre les deux objets les plus éloignés de chaque cluster. Distance Moyenne (Average Linkage): Moyenne de toutes les distances possibles entre paires d'objets des deux clusters. Coûteux. Centroïde: Distance entre les centroïdes (moyennes) des clusters. Critère de Ward: Basé sur l'inertie, tend à créer des clusters de cardinalités (nombre d'objets) similaires.
- Le choix du critère de lien entraîne des résultats de clustering différents même avec les mêmes données.

## Exemples d'Utilisation
- Contrainte: La complexité quadratique de la matrice de distance limite son application aux grands volumes de données. Utilisé quand le nombre d'objets est "acceptable".
- Applications: Bioinformatique: Regroupement de données d'expression de gènes. Neurosciences: Clustering de représentations d'images issues d'IRM cérébrales pour étudier la représentation abstraite et visuelle. Analyse comportementale: Clustering de séquences d'activités de chirurgiens (seniors vs. juniors) pour identifier des schémas opératoires.

## Évaluation du Clustering
- Différence avec la classification: Pas toujours de vérité terrain (labels de classe) pour vérifier la qualité des clusters.
- Deux approches principales: Critères Internes: Basés uniquement sur la qualité intrinsèque des clusters. Mesurent la compacité (objets proches au sein d'un cluster) et la séparation (clusters distincts les uns des autres) dans l'espace des données. Exemples: Coefficient de Silhouette, Dunn Index. Critères Externes: Comparent la partition obtenue par l'algorithme à un clustering ou à des classes préexistantes (si disponibles). Gèrent l'absence de correspondance directe entre les numéros de cluster (ex: Cluster 1) et les noms de classes (ex: Iris setosa), ainsi que la possibilité d'un nombre de clusters différent. Exemples: Rand Index, Adjusted Rand Index.

## Conclusion : Avantages et Inconvénients du Clustering Hiérarchique Ascendant
- Avantages: Simplicité et Intuitivité: Facile à comprendre (processus itératif de regroupement). Flexibilité du nombre de classes: Le nombre de clusters n'est pas un paramètre initial; il est choisi après la construction du dendrogramme par une simple coupe. Visualisation intuitive: Le dendrogramme aide à orienter le choix du nombre de clusters.
- Inconvénients: Coût de la matrice de distance: La complexité quadratique O(n^2) (où n est le nombre d'objets) rend l'algorithme inadapté aux très grands volumes de données. Décisions irréversibles: Les fusions de clusters ne sont jamais remises en question au cours de l'algorithme (approche gloutonne). Variabilité: Les résultats dépendent fortement du critère de lien choisi (single, complete, average, Ward, etc.), ce qui peut rendre difficile le choix du "meilleur" critère.

# Decision Tree

## Qu'est-ce qu'un Arbre de Décision ?
- Définition: Une structure d'arbre où chaque composant a une fonction spécifique.
- Structure: Nœuds: Représentent des tests sur les attributs du jeu de données. Branches: Correspondent aux valeurs possibles que peut prendre un attribut. Feuilles: Représentent une classe (pour la classification) ou une valeur continue (pour la régression).

## Jeu de Données Exemple: Météo et Jeu
- Contexte: Prédiction de la participation à un jeu (ex: golf/tennis) en fonction des conditions météorologiques.
- Attributs (exemples): État du ciel (Sky), Température (Temperature), Humidité (Humidity), Vent (Wind).
- Classe à prédire: Jouer (Yes/No).
- Processus d'inférence: Parcourir l'arbre en fonction des conditions météo pour arriver à une feuille (décision Yes/No).

## Construction de l'Arbre de Décision: Algorithme ID3
- Algorithme principal: ID3 (Iterative Dichotomiser 3)
- Proposé par: Quinlan en 1986.
- Approche: top-down (descendante), itérative.
- Concepts clés: Entropie: Mesure l'incertitude d'un ensemble de données. Maximale quand il y a équiprobabilité (ex: 0.5 probabilité pour deux événements, entropie = 1). Nulle quand il n'y a aucune incertitude (ex: 100% de "Yes"). Gain d'information: Mesure la réduction de l'incertitude sur la classification suite à la connaissance d'un attribut. L'objectif est de trouver l'attribut qui offre le plus grand gain d'information à chaque étape.
- Processus de construction (simplifié): 1. Calculer l'entropie de la classe de la population totale (entropie initiale). 2. Pour chaque attribut, calculer le gain d'information qu'il apporte. 3. Choisir l'attribut avec le gain d'information le plus élevé comme nœud racine (ou nœud interne). 4. Répéter le processus récursivement pour chaque branche, jusqu'à ce que les feuilles soient pures (toutes de la même classe) ou qu'un critère d'arrêt soit atteint.
- Exemple de calcul initial (9 Yes, 5 No): Entropie de la population ≈ 0.94.

## Gestion des Valeurs Numériques
- Problème: Impossible de créer une branche pour chaque valeur continue possible (ex: température en degrés Fahrenheit, humidité en pourcentage).
- Solution: Discrétisation. Transformer une variable continue en variable discrète (ex: température < 70 = "froid", 70-75 = "doux", > 75 = "chaud"). Peut être effectuée en pré-traitement des données. Certains algorithmes (ex: C4.5) intègrent la discrétisation et choisissent les seuils de coupe optimaux pendant la construction de l'arbre.

## Surapprentissage (Overfitting)
- Définition: Le modèle adhère trop aux données d'entraînement, ce qui le rend incapable de bien généraliser sur de nouvelles données.
- Causes spécifiques aux arbres de décision: Profondeur excessive de l'arbre. Trop de petites coupes spécialisées sur les données d'entraînement. Données d'entraînement insuffisantes. Absence d'élagage (pruning).
- Stratégies d'Élagage (Pruning) pour l'éviter: Supprimer les nœuds qui n'améliorent pas significativement les performances. Ajouter des scores de pénalité pour les arbres trop complexes. Utiliser la théorie MDL (Minimum Description Length) pour trouver la description minimale optimale. Techniques d'élagage basées sur les erreurs. Contrôler la profondeur maximale de l'arbre. Pré-élagage (contraintes pendant la construction) vs. Post-élagage (modification de l'arbre après construction).

## Forêts Aléatoires (Random Forests)
- Concept: Une collection d'arbres de décision (ensemble learning). L'idée est de construire plusieurs arbres et de combiner leurs décisions/prédictions pour obtenir une prédiction finale.
- Avantages: Très performantes et largement utilisées. Peuvent surpasser l'apprentissage profond sur certaines applications spécifiques.
- Diversité des arbres: Essentielle car la construction d'un arbre unique (ex: avec ID3) est déterministe.
- Méthodes pour introduire la diversité: Utiliser différents algorithmes de construction. Appliquer différentes méthodes d'élagage. Échantillonnage des données: Échantillonnage des instances (ne pas utiliser toutes les données pour chaque arbre). Échantillonnage des attributs (ne pas utiliser tous les attributs pour chaque arbre).

## Avantages et Inconvénients des Arbres de Décision
- Avantages: Interprétables et explicables (très visuels). Faible préparation des données (normalisation souvent intégrée). Gestion des données numériques et catégorielles. Capacité à représenter des frontières de classe non linéaires.
- Désavantages: Surapprentissage fréquent (sans élagage). Instabilité: Les petits changements dans les données peuvent entraîner des modifications significatives de l'arbre. Complexité de mise à jour: Nécessite souvent une reconstruction complète de l'arbre lors de l'ajout de nouvelles données. Biais: Peut survaloriser des attributs avec de nombreuses valeurs possibles. Optimum local: L'algorithme glouton de construction ne garantit pas l'optimum global de l'arbre.

## Conclusion
Les arbres de décision, bien que potentiellement instables et sujets au surapprentissage, restent un modèle très pertinent, surtout pour les petits volumes de données. Les Random Forests, leurs dérivés, sont des algorithmes extrêmement performants et très utilisés dans diverses applications.
