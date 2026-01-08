#!/usr/bin/env python3
# Auto-generated single file for Numworks calculator
# All functionality in one file for Numworks compatibility

import re
import math

# Utilities functions

def strip_markdown(text):
    """Strip basic markdown syntax from text."""
    # Remove headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove bold
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    # Remove italic
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove lists
    text = re.sub(r"^[\s]*[-\*\+]\s*", "", text, flags=re.MULTILINE)
    return text

def display_text(text, max_lines=20):
    """Display text in chunks for limited screen."""
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        print(chunk)
        if i + max_lines < len(lines):
            input("Press Enter to continue...")

# Calculators functions

def bayes_calculator():
    """Interactive Bayes theorem calculator."""
    print("Bayes Calculator")
    try:
        prior = float(input("Enter prior probability P(H): "))
        likelihood = float(input("Enter likelihood P(E|H): "))
        evidence = float(input("Enter evidence P(E): "))
        posterior = (prior * likelihood) / evidence
        print("Posterior probability P(H|E): {}".format(posterior))
    except ValueError:
        print("Invalid input. Please enter numbers.")
    except ZeroDivisionError:
        print("Error: Evidence P(E) cannot be zero.")

def euclidean_distance_calculator():
    """Interactive Euclidean distance calculator."""
    print("Euclidean Distance Calculator")
    try:
        n = int(input("Enter number of dimensions: "))
        point1 = []
        point2 = []
        for i in range(n):
            p1 = float(input("Enter coordinate {} for point 1: ".format(i+1)))
            p2 = float(input("Enter coordinate {} for point 2: ".format(i+1)))
            point1.append(p1)
            point2.append(p2)
        distance = sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
        print("Euclidean distance: {}".format(distance))
    except ValueError:
        print("Invalid input. Please enter numbers.")

def manhattan_distance_calculator():
    """Interactive Manhattan distance calculator."""
    print("Manhattan Distance Calculator")
    try:
        n = int(input("Enter number of dimensions: "))
        point1 = []
        point2 = []
        for i in range(n):
            p1 = float(input("Enter coordinate {} for point 1: ".format(i+1)))
            p2 = float(input("Enter coordinate {} for point 2: ".format(i+1)))
            point1.append(p1)
            point2.append(p2)
        distance = sum(abs(a - b) for a, b in zip(point1, point2))
        print("Manhattan distance: {}".format(distance))
    except ValueError:
        print("Invalid input. Please enter numbers.")

def entropy_calculator():
    """Interactive entropy calculator."""
    print("Entropy Calculator")
    try:
        probs = []
        while True:
            prob = input("Enter probability (or 'done'): ")
            if prob.lower() == "done":
                break
            probs.append(float(prob))
        if not probs:
            print("No probabilities entered.")
            return
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        print("Entropy: {}".format(entropy))
    except ValueError:
        print("Invalid input. Please enter numbers.")

def gini_calculator():
    """Interactive Gini impurity calculator."""
    print("Gini Impurity Calculator")
    try:
        probs = []
        while True:
            prob = input("Enter class probability (or 'done'): ")
            if prob.lower() == "done":
                break
            probs.append(float(prob))
        if not probs:
            print("No probabilities entered.")
            return
        gini = 1 - sum(p ** 2 for p in probs)
        print("Gini impurity: {}".format(gini))
    except ValueError:
        print("Invalid input. Please enter numbers.")

def information_gain_calculator():
    """Interactive information gain calculator."""
    print("Information Gain Calculator")
    try:
        parent_entropy = float(input("Enter parent entropy: "))
        n_children = int(input("Enter number of child nodes: "))
        total_samples = int(input("Enter total samples in parent: "))
        weighted_child_entropy = 0
        for i in range(n_children):
            samples = int(input("Enter samples in child {}: ".format(i+1)))
            entropy = float(input("Enter entropy of child {}: ".format(i+1)))
            weighted_child_entropy += (samples / total_samples) * entropy
        info_gain = parent_entropy - weighted_child_entropy
        print("Information Gain: {}".format(info_gain))
    except ValueError:
        print("Invalid input. Please enter numbers.")
    except ZeroDivisionError:
        print("Error: Total samples cannot be zero.")

def execute_math():
    """Simple math expression evaluator."""
    print("Math Executor")
    print("Available functions: sqrt, sin, cos, tan, log, log2,")
    print("log10, exp, pow, abs")
    expr = input("Enter math expression: ")
    try:
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "exp": math.exp,
            "pow": pow,
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        print("Result: {}".format(result))
    except Exception as e:
        print("Invalid expression: {}".format(e))

# Content data

TOPIC_CONTENTS = {
    "cours": {
        "cours/cours.md": [
            "# Synthèse Data Mining - Cours Germain Forestier",
            "",
            "## 1. Introduction au Data Mining",
            "Le Data Mining (fouille de données) découvre modèles, relations et connaissances dans grands ensembles de données. Combine statistiques, IA et Machine Learning.",
            "",
            "### Processus CRISP-DM",
            "1. **Business Understanding**: Définir objectifs métier et critères de succès.",
            "2. **Data Understanding**: Collecter, décrire et explorer données (qualité).",
            "3. **Data Preparation**: Nettoyage, fusion, transformation (80% du temps).",
            "4. **Modeling**: Sélectionner et appliquer algorithmes.",
            "5. **Evaluation**: Vérifier si modèle répond aux objectifs business.",
            "6. **Deployment**: Mise en production et maintenance.",
            "",
            "### Prétraitement et Qualité des Données",
            "- **Structurées**: Tables (CSV, SQL). **Non structurées**: Texte, images, vidéos.",
            "- **Nettoyage**: Gérer valeurs manquantes (suppression ou imputation par moyenne/médiane/mode), détecter outliers.",
            "- **Transformation**: Normalisation Min-Max $[0, 1]$ ou Z-score (moy 0, écart-type 1).",
            "- **EDA**: Visualisation (Histogrammes: distribution, Boxplots: dispersion/outliers, Nuages: relations) pour comprendre données.",
            "",
            "### Concepts Statistiques et ML",
            "- **Supervisé**: Données étiquetées (Classification: discret, Régression: continu).",
            "- **Applis**: Contrôle qualité, prédiction pannes, météo, rendements.",
            "",
            "### Data Analysis vs. Data Mining vs. Machine Learning",
            "- **Data Analysis**: Compréhension données via méthodes statistiques.",
            "- **Data Mining**: Utilise algos pour trouver modèles/prédire tendances. Inclut prétraitement, EDA, interprétation.",
            "- **Machine Learning (ML)**: Sous-ensemble IA, entraîne modèles. Utilisé dans Data Mining.",
            "",
            "### Intersections",
            "- IA, ML, Big Data, Rech. Info, Reco. Formes.",
            "- Text Mining, BDD, Anal. Prédictive, NLP, Vision (OCR).",
            "",
            "### Relation AI, ML, Deep Learning",
            "Deep Learning $\\subset$ Machine Learning $\\subset$ Intelligence Artificielle.",
            "",
            "### \"Tendance\" et Data Scientist",
            "- Termes ML, Big Data, Data Science, AI souvent interchangeables.",
            "- Data Scientist: Profil transdisciplinaire très demandé (AI engineer/scientist).",
            "",
            "## 2. Types et Sources de Données",
            "### Structurées vs. Non Structurées",
            "- **Structurées**: Format défini (tables, CSV). Ex: Iris.",
            "- **Non Structurées**: Pas de forme spécifique (texte, vidéos, images). Traitement spécialisé.",
            "- **Problème**: Classes mal balancées (ex: maladies rares).",
            "",
            "### Formats Courants",
            "- **CSV**: Texte brut, sép. virgules.",
            "- **JSON**: Texte, lisible, web.",
            "- **XML**: Balisage, auto-descriptif.",
            "- **Autres**: YAML, HDF5.",
            "",
            "### Sources",
            "- **BDD**: Structurées (relationnelles, doc, clé-valeur).",
            "- **Web Scraping**: Extraction sites web.",
            "- **APIs**: Protocoles interaction apps/données.",
            "",
            "## 3. Prétraitement des Données",
            "### Données Propres et Qualité",
            "- **Propres**: Sans erreurs/incohérences, améliore précision, facilite intégration.",
            "- **Qualité**: Normes (précision, complétude, fiabilité), soutient décision.",
            "- **Mauvaise qualité**: Conclusions incorrectes, coûts up, confiance down.",
            "- **Note**: Prép = bcp de temps. \"Cleaned dataset\" nécessaire.",
            "",
            "### Techniques Nettoyage",
            "- **Val. manquantes**: Supprimer, Imputer (moy, med, mode), Prédire (ML).",
            "- **Transformation**: Norm/scale, conversion types, encodage catégoriel.",
            "- **Outliers**: Identifier/gérer. Prudence (info importante). Pertinents dans train/test.",
            "- **Validation**: Vérifier précision avec sources externes.",
            "",
            "## 4. Analyse Exploratoire (EDA)",
            "### But",
            "- **Comprendre struct**: Forme, tendance centrale, dispersion, relations.",
            "- **Identifier modèles**: Tendances, clusters, outliers.",
            "- **Communiquer**: Visualisations.",
            "",
            "### Visualisation",
            "- **Barres**: Fréq/catégorie.",
            "- **Nuages points**: Relation 2 vars continues.",
            "- **Linéaires**: Tendances temps.",
            "- **Histogrammes**: Distrib var continue.",
            "- **Boxplots**: Tend centrale/variabilité.",
            "",
            "## 5. Outils et Logiciels",
            "### Populaires",
            "- **Python**: Généraliste, écosystème riche (pandas, matplotlib, scikit-learn, NumPy).",
            "- **R**: Calcul stat.",
            "- **SQL**: Requêtes BDD.",
            "- **Excel**: Tableur.",
            "",
            "### Bibliothèques",
            "- **Pandas**: Manip/analyse (dataframe).",
            "- **NumPy**: Calcul num, matrices.",
            "- **Matplotlib**: Visu.",
            "- **Scikit-learn**: ML (classif, reg, clust, preproc).",
            "",
            "## 6. Concepts Base Stat et ML",
            "### Descriptives vs. Inférentielles",
            "- **Descriptives**: Résumé (moy, med, mode, écart-type). Aperçu.",
            "- **Inférentielles**: Préd/inférences sur pop via échantillon. Conclusions générales.",
            "",
            "### Supervisé vs. Non Supervisé",
            "- **Supervisé**: Données étiquetées. Apprend fct entrée->sortie (classif, reg). Prédire futur.",
            "- **Non Supervisé**: Non étiquetées. Ident modèles/struct (clustering, réduction dim). Description.",
            "",
            "### Biais-Variance",
            "- **Biais**: Erreur simplif excessive -> sous-apprentissage (underfitting).",
            "- **Variance**: Sensibilité fluctuations -> sur-apprentissage (overfitting).",
            "- **Compromis**: Équilibrer pour min erreur totale. Précision et interprétabilité.",
            "",
            "## 7. Processus CRISP-DM",
            "1. **Business Understanding**: Obj métier. Alignement, métriques, ressources.",
            "2. **Data Understanding**: Collecte, descr, explo, qualité.",
            "3. **Data Preparation**: Clean, transform, integrate, select.",
            "4. **Modeling**: Sélec techniques, build, eval.",
            "5. **Evaluation**: Qualité modèle, révision.",
            "6. **Deployment**: Plan déploiement, maint, surv.",
            "",
            "## 8. Big Data et Scalabilité",
            "### 5 V",
            "- **Volume**: Taille (To, Po).",
            "- **Vélocité**: Vitesse gén/traitement.",
            "- **Variété**: Types (struct, non-struct).",
            "- **Véracité**: Qualité/fiabilité.",
            "- **Valeur**: Potentiel dérivé.",
            "- **Tech**: Hadoop, Spark, NoSQL.",
            "",
            "### Défis",
            "- Stockage, Traitement (//), Intégration, Qualité, Sécurité, Analyse, Scalabilité, Coût.",
            "- **Pb**: Algos durs sur énormes datasets -> sous-ensembles.",
            "",
            "## 9. Exemples Concrets",
            "### Applications",
            "- **Santé**: Épidémies, traitements.",
            "- **Finance**: Fraude, risques.",
            "- **Retail**: Recom, prix, stocks (Walmart).",
            "- **Fab**: Qualité, process.",
            "- **Transport**: Trafic, itin (GMap), maint (GE).",
            "- **Énergie**: Prév demande.",
            "- **Divert**: Recom (Netflix).",
            "- **Gouv**: Séc publique, météo (NWS).",
            "",
            "### Succès",
            "Netflix, AmEx, Walmart, GE Aviation, Google Maps, Watson Santé, NWS, LinkedIn.",
            "",
            "# Bayes",
            "## Résumé",
            "Classifieur proba (Th. Bayes + indép attributs). Trouve classe la plus probable. Utile classif/texte.",
            "",
            "## 1. Intro",
            "- Utilise probas pour classif.",
            "- Affecte proba à chaque hypothèse (classe).",
            "- Obs train modifient distrib proba.",
            "- Cherche hyp la plus probable sachant instance.",
            "- Base: Proba cond + Bayes. Hypothèse indép attributs.",
            "",
            "## 2. Rappels Proba",
            "- P(A) $\\in [0,1]$.",
            "- Indép: $P(A \\cap B) = P(A)P(B)$.",
            "- Cond: $P(A|B) = P(A \\cap B) / P(B)$.",
            "",
            "## 3. Th. Bayes",
            "$$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$",
            "",
            "## 4. Application Classification",
            "- Calc $P(C_k|X)$ pour chaque classe.",
            "- $P(C|Desc) = \\frac{P(Desc|C)P(C)}{P(Desc)}$",
            "- **Postériori**: $P(C|Desc)$. **Vraisemblance**: $P(Desc|C)$. **Priori**: $P(C)$.",
            "- Estim via train: $P(C)$=prop classe, $P(Desc|C)$=freq desc ds classe.",
            "",
            "### 4.2 MAP (Maximum A Posteriori)",
            "- Choisir $C_k$ max $P(C_k|Desc)$.",
            "- $P(Desc)$ constant -> $\\text{argmax}_k [P(Desc|C_k)P(C_k)]$.",
            "",
            "## 5. Naive Bayes (Indép)",
            "- Hyp: Attributs indép sachant classe.",
            "- $P(Desc|C) = \\prod P(a_i|C)$.",
            "- **Final**: $\\text{argmax}_k [P(C_k) \\times \\prod P(a_i|C_k)]$.",
            "- Discret: $P(a_i|C_k)$ = freq val $a_i$ ds $C_k$.",
            "",
            "## 6. Ex: Jeu",
            "- Train: Genre, Plat, Budg, Pop(O/N). X: RPG, PC, Med.",
            "- **Yes**: $P(Y)=3/5$. Prod $P(attr|Y)=0.0444$.",
            "- **No**: $P(N)=2/5$. Prod $P(attr|N)=0.05$.",
            "- **Pred**: No ($0.05 > 0.0444$).",
            "- **Norm**: Somme=0.0944. P(Y)=47%, P(N)=53%.",
            "",
            "## 7. Petites Probas",
            "- Prod $< 1$ -> valeurs très petites.",
            "- **Log**: $\\log(ab)=\\log a+\\log b$. Somme évite underflow.",
            "",
            "## 8. Numérique",
            "- Estimer distrib (Gaussienne: $\\mu, \\sigma$ par classe).",
            "",
            "## 9. Analyse Texte",
            "- Classif (spam).",
            "- **Bag-of-Words**: Mot=attr (présence/fréq).",
            "",
            "## 10. Avantages/Inconvénients",
            "- **+**: Simple, efficace, interprétable, perf petits data, scale.",
            "- **-**: Indép attr (rare), Zéro Proba (lissage Laplace), Modèles complexes.",
            "",
            "# Clustering",
            "## Intro",
            "- Non supervisé. Groupes (clusters).",
            "- Max similarité intra, Min similarité inter.",
            "",
            "## Distance",
            "- Mesure ressemblance.",
            "- **Euclidienne**: $\\sqrt{\\sum \\text{diff}^2}$. **Norm** (Min-Max) cruciale.",
            "- **Matrice Dist**: $2 \\times 2$, complexité quadratique.",
            "",
            "## Hiérarchique Ascendant",
            "1. **Init**: Chq obj = clust.",
            "2. **Iter**: Fusion 2 + proches.",
            "3. **Fin**: 1 seul clust.",
            "- **Dendrogramme**: Arbre fusions. Coupe -> nb clusters.",
            "- **Linkage**: Min (Single, chaînage), Max (Complete), Moy (Average), Ward (inertie, équil).",
            "",
            "## Eval",
            "- Pas vérité terrain.",
            "- **Interne**: Qual intrinsèque (Silhouette, Dunn).",
            "- **Externe**: Comp existant (Rand, Adj Rand).",
            "",
            "## Concl",
            "- **+**: Simple, nb classes flex, visu.",
            "- **-**: $O(n^2)$ lourd, glouton (irrév), sensib lien.",
            "",
            "# Decision Tree",
            "## Quoi?",
            "- Arbre: Nœuds (tests), Branches (val), Feuilles (classe).",
            "",
            "## Construction (ID3)",
            "- Top-down, iter.",
            "- **Entropie**: Incertitude. **Gain Info**: Réduc incert.",
            "- Choisir attr max gain -> Nœud.",
            "",
            "## Numérique",
            "- **Discrétisation**: Seuils (ex: $<70$). C4.5 le fait.",
            "",
            "## Surapprentissage (Overfitting)",
            "- Modèle trop collé train.",
            "- **Pruning**: Suppr nœuds, pénalité compl, prof max.",
            "",
            "## Random Forests",
            "- Ensemble d'arbres. Vote.",
            "- **Div**: Algos diff, éch data/attr.",
            "- **+**: Perf élevée.",
            "",
            "## Pros/Cons",
            "- **+**: Interprétable, peu prép, num/cat, non-lin.",
            "- **-**: Overfit, instable, biais, opt local.",
            "",
            "# Kmeans",
            "## Principes",
            "- Non sup, Partition (K), K fixé.",
            "- Clusters compacts. Means = Moyennes.",
            "",
            "## Algo",
            "1. Select K.",
            "2. Init K centres.",
            "3. Affect obj centre + proche.",
            "4. Recalc centres (moy).",
            "5. Répéter tant que bouge.",
            "",
            "## Aspects",
            "- **K**: Délicat. Intuition, Coude, Silhouette.",
            "- **Dist**: Euclid, Manh.",
            "- **Init**: Sensible (opt local). K-means++.",
            "- **Outliers**: Influencent moy. K-medoids.",
            "",
            "## Variantes",
            "- **K-medoids** (obj réel), **Fuzzy** (degré), **K-means++** (init).",
            "",
            "## Pros/Cons",
            "- **+**: Rapide, Interpr, Simple.",
            "- **-**: Choix K, Forme sphérique, Sensib init/outliers.",
            "",
            "# Knn",
            "## Takeaway",
            "Lazy, classif/reg via maj/moy k voisins. Simple mais lourd.",
            "",
            "## Intro",
            "- Instance-based/Lazy. Classif new via maj k voisins. 1-NN: + proche.",
            "",
            "## Concepts",
            "- **Dist**: Euclid, Manh.",
            "- **Lazy**: Pas modèle, tout train = modèle.",
            "",
            "## Pros/Cons",
            "- **+**: Simple, Polyv, Rapide (no train), Non-param, Update, Explic.",
            "- **-**: Coût ($O(N)$), Bruit (pt k), Choix k (impair), Norm indispensable.",
            "",
            "## Détails",
            "- **Ex**: Iris.",
            "- **Pondér**: Vote maj ou dist ($1/d^2$).",
            "- **Cat**: One-Hot.",
            "",
            "# Neural Network",
            "## Hist",
            "- 60s: Perceptron. XOR imp -> AI Winter.",
            "- 74: Backprop (Werbos).",
            "- 90s: CNN (LeCun).",
            "- 2012: AlexNet (DL boom).",
            "",
            "## Perceptron",
            "- Entrées+Biais -> Poids -> Somme -> Act -> Sortie.",
            "- **Apprent**: Corr err. $W_{new} = W + \\eta(C-O)X$.",
            "",
            "## Deep Learning",
            "- Profond, bcp params. Non-lin.",
            "- **Archis**: LeNet-5, AlexNet, Inception, VGG16, Transformers.",
            "- Facteurs: Profondeur, GPU.",
            "",
            "# Time Series",
            "## Intro",
            "- Séq ordonnée tps. Ordre compte.",
            "- Uni/Multi. Decomp (Trend, Sais, Cyc, Bruit).",
            "",
            "## Feat Extract",
            "- Stats ($\\mu, \\sigma$, Z-Norm). Auto-Corr. Fourier. catch22.",
            "",
            "## Analysis",
            "- **Forec**: MA (err), AR (val).",
            "- **Imput**: Moy, Med, Ffill, Bfill.",
            "- **Classif**: KNN (Euclid), DTW (align temp).",
            "- **Unequal**: Pad, Trunc, DTW.",
            "- **Dim Red**: PCA, PAA. Symb: SAX, SFA.",
            "",
            "# Wooclap",
            "**DM**: Patterns. **Struct**: CSV. **Preproc**: Clean. **EDA**: Understand. **Miss**: Mean. **Dim**: PCA. **BDD**: SQL. **Unsup**: Clust. **Big**: Insights. **Qual**: Eval. **Unstr**: No form. **Pred**: ML. **Ver**: Qual. **Visu**: Hist.",
            "**KNN**: Inst-based, Euclid, Cost, Noise(sm k), Scale, W-vote, Lazy, OneHot, CV, Med, Norm, Odd k.",
            "**DT**: Class/Reg, Test node, ID3, Entr, Overfit, Prun, CCP, InfoGain, RF, Num/Cat, MDL, Discr.",
            "**Bayes**: Cond, Indep, P(A), P(AB), MAP, Word, ZeroProb, Sm data, Log, Gauss, Spam.",
            "**Hie Clust**: Grps, Unsup, Euclid, Scale, Dendro, Sgl(noise), Ward(bal), Cut, Silh(int), Rand(ext), Cost, Centr.",
            "**KMeans**: Min var, K rand, Mean, K++, Conv, Euclid, Outlier, Elbow, Spher, Medoids, Unsup, Silh.",
            "**NN**: Rosenb, Nonlin, Werbos, Heaviside, LeNet, AlexNet, Incept, VGG, ErrCorr, 1x1, ReLU, DL, Nonlin bnd.",
            "**TS**: Order, ZNorm, Forec, DTW, Trend, Ffill, Four, Var, AR."
        ]
    }
}

TOPIC_DISPLAY_NAMES = {{
    "cours": "Cours"
}}

# Content functions

def get_all_topics():
    """Return list of all available topics."""
    return sorted(TOPIC_CONTENTS.keys())

def get_topic_content(topic):
    """Get content dict for a topic."""
    return TOPIC_CONTENTS.get(topic, {{}})

def get_files(topic):
    """Return list of files in a topic."""
    return sorted(get_topic_content(topic).keys())

def get_content(topic, file_key):
    """Get content for a specific file in a topic."""
    lines = get_topic_content(topic).get(file_key, [])
    return "\n".join(lines)

def search_topic(topic, query):
    """Search for query in a topic."""
    results = []
    for file_key, content in get_topic_content(topic).items():
        if query.lower() in "\n".join(content).lower():
            results.append(file_key)
    return results

def search_all_topics(query):
    """Search across all topics."""
    results = []
    for topic in get_all_topics():
        for file_key in search_topic(topic, query):
            results.append((topic, file_key))
    return results

# Menu functions

def view_topic_menu():
    """Display topic selection menu."""
    print("\nAvailable Topics:")
    topics = get_all_topics()
    for i, topic in enumerate(topics, 1):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        print("{}. {}".format(i, display_name))
    print("0. Back to main menu")
    choice = input("\nSelect a topic: ")
    if choice == "0":
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(topics):
            topic_key = topics[idx]
            display_name = TOPIC_DISPLAY_NAMES.get(topic_key, topic_key)
            view_topic_files(topic_key, display_name)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")

def view_topic_files(topic_key, display_name):
    """View files within a specific topic."""
    files = get_files(topic_key)
    if not files:
        print("No files found in {}.".format(display_name))
        return
    while True:
        print("\n{} - Available Files:".format(display_name))
        for i, file in enumerate(files, 1):
            print("{}. {}".format(i, file))
        print("0. Back to topics")
        choice = input("\nSelect a file to view: ")
        if choice == "0":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                content = get_content(topic_key, files[idx])
                stripped = strip_markdown(content)
                print("\n--- {} ---\n".format(files[idx]))
                display_text(stripped)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")

def search_all_topics_menu():
    """Search across all topics."""
    query = input("Enter search query: ")
    if not query:
        return
    results = search_all_topics(query)
    if not results:
        print("No matches found.")
        return
    print("\nFound {} matches:".format(len(results)))
    for i, (topic, file_key) in enumerate(results, 1):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        print("{}. [{}] {}".format(i, display_name, file_key))
    choice = input("\nEnter number to view (or 'q' to quit): ")
    if choice.lower() == "q":
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            topic, file_key = results[idx]
            content = get_content(topic, file_key)
            stripped = strip_markdown(content)
            print("\n--- {} ---\n".format(file_key))
            display_text(stripped)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")

def calculators_menu():
    """Display calculators menu."""
    while True:
        print("\nCalculators:")
        print("1. Bayes Calculator")
        print("2. Euclidean Distance Calculator")
        print("3. Manhattan Distance Calculator")
        print("4. Entropy Calculator")
        print("5. Gini Impurity Calculator")
        print("6. Information Gain Calculator")
        print("7. Math Expression Evaluator")
        print("0. Back to main menu")
        choice = input("\nSelect a calculator: ")
        if choice == "0":
            return
        elif choice == "1":
            bayes_calculator()
        elif choice == "2":
            euclidean_distance_calculator()
        elif choice == "3":
            manhattan_distance_calculator()
        elif choice == "4":
            entropy_calculator()
        elif choice == "5":
            gini_calculator()
        elif choice == "6":
            information_gain_calculator()
        elif choice == "7":
            execute_math()
        else:
            print("Invalid choice.")

def main():
    """Main menu."""
    while True:
        print("\n" + "=" * 50)
        print("Data Mining Course Viewer and Calculator")
        print("=" * 50)
        print("1. Browse Topics")
        print("2. Search All Content")
        print("3. Calculators")
        print("4. Quit")
        choice = input("\nChoose an option: ")
        if choice == "1":
            view_topic_menu()
        elif choice == "2":
            search_all_topics_menu()
        elif choice == "3":
            calculators_menu()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
