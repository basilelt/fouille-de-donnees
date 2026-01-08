# Auto-generated content module for Neural Network
# Contains embedded markdown content for this topic

TOPIC_NAME = "Neural Network"
TOPIC_KEY = "neural-network"

CONTENT = {
    "neural-network/1.md": """# Notes de Cours : Introduction aux Réseaux de Neurones

Ce cours de fouille de données est dédié aux réseaux de neurones, couvrant leur historique, les principes fondamentaux du Perceptron, l'algorithme d'apprentissage, un exercice pratique et une introduction au Deep Learning.

## 1. Historique des Réseaux de Neurones

- **Années 1960**
  - Rosenblatt propose le Perceptron, un classifieur binaire.
  - Minsky et Papert mettent en évidence l'incapacité des Perceptrons simples à résoudre des problèmes non linéaires (ex: problème XOR).
  - "AI Winter": Période de perte de confiance due à cette limitation.

- **1974**
  - Werbos propose l'algorithme de Rétropropagation du gradient (Backpropagation), permettant de résoudre les problèmes non linéaires en utilisant des réseaux multicouches.
  - L'algorithme est réutilisé et popularisé par Mel, Hinton et Williams pour l'entraînement des réseaux multicouches.

- **Années 1990**
  - Apparition des CNN (Convolutional Neural Networks) pour la classification d'images, proposés par Yann LeCun.
  - Yann LeCun reçoit le Prix Turing (équivalent Nobel en informatique) pour ses travaux sur les CNN.

## 2. Le Perceptron : Bases et Fonctionnement

- **Définition**: Un neurone est une unité de calcul qui prend des entrées, effectue un calcul et renvoie une sortie.
- **Entrées**: n entrées (x1 à xn) et un biais x0 toujours égal à 1.
- **Paramètres**: Les poids (weights) W0 à WN sont les valeurs apprises pendant l'entraînement.
- **Calcul de la sortie (version simple)**:
  1. Calcul d'une somme pondérée des entrées et des poids (∑(xi * Wi)).
  2. Cette somme est passée à une fonction d'activation qui calcule la sortie finale du neurone.
- **Limitation**: Le perceptron de base modélise des décisions linéaires et ne peut pas résoudre des problèmes non linéaires.
- **Représentation schématique**: Entrées (x0...xn) -> Poids (W0...Wn) -> Somme pondérée -> Fonction d'activation -> Sortie.

## 3. Algorithme d'Apprentissage du Perceptron : Correction par Erreur

- **Principe**: Ajuster les poids du réseau pour maximiser le taux de bonnes réponses.
- **Étapes**:
  1. Initialiser le perceptron et ses poids à des valeurs arbitraires.
  2. Pour chaque exemple d'entraînement:
     - Présenter l'exemple au réseau.
     - Calculer la sortie.
     - Comparer la sortie avec la classe attendue.
     - Si la classification est incorrecte, ajuster les poids.
  3. L'algorithme s'arrête lorsque tous les exemples sont correctement classés et qu'aucun changement de poids n'est nécessaire (stabilité).
- **Formule de mise à jour des poids**:
  $W_{\\text{nouveau}} = W_{\\text{précédent}} + \\eta \\times (C - O) \\times X_{\\text{entrée}}$
- **η**: Taux d'apprentissage (non explicitement mentionné mais implicite dans le coefficient multiplicateur, souvent appelé alpha).
- **C**: Classe attendue (cible).
- **O**: Sortie calculée par le réseau.
- **X_entrée**: Valeur de l'entrée correspondante.
- **Si C = O, (C - O) est 0, donc les poids ne sont pas modifiés.**
- **Note**: Modifier les poids peut altérer la classification correcte d'exemples précédemment bien classés, nécessitant plusieurs passes sur l'ensemble des exemples.

## 4. Exercice Pratique : Apprentissage du "OU Booléen"

- **Objectif**: Entraîner un perceptron à apprendre la fonction logique OU.
- **Entrées**: x0=1 (biais), x1 et x2 (binaires: 0 ou 1).
- **Sortie attendue**: x1 OU x2.
- **Processus**:
  1. Initialiser les poids arbitrairement (ex: 0, -1, 1 pour W0, W1, W2).
  2. Parcourir séquentiellement les exemples du tableau d'entraînement.
  3. Pour chaque exemple, calculer la sortie et mettre à jour les poids si nécessaire.
  4. Repasser sur l'ensemble des exemples si des poids ont été modifiés, jusqu'à ce que les poids soient stables et que tous les exemples soient bien classés.
- **Résultat stable (ex)**: Poids finaux de 0, 1, 1 pour W0, W1, W2 (pour le ou booléen).

## 5. Deep Learning (Apprentissage Profond)

- **Définition**: Terme "à la mode" désignant des réseaux de neurones profonds, c'est-à-dire des réseaux avec plusieurs couches de neurones.
- **Caractéristiques**: Contiennent de nombreuses couches et un très grand nombre de paramètres (parfois des milliards dans les réseaux modernes).
- **Principe fondamental**: Le même que celui du perceptron simple (unités de calcul qui prennent, calculent et transmettent de l'information), mais avec une complexité accrue due à la profondeur.
- **Succès**: Capacité à entraîner des réseaux multicouches pour apprendre des décisions non linéaires complexes.
- **Exemples d'architectures notables**:
  - **LeNet-5 (1998)**:
    - Conçu pour la reconnaissance de chiffres manuscrits (dataset MNIST).
    - Contenait environ 60 000 paramètres.
    - Structure typique: couches de convolution, couches de sous-échantillonnage, couche entièrement connectée, distribution de probabilité en sortie.
  - **AlexNet (2012)**:
    - A révolutionné la classification d'images sur le benchmark ImageNet.
    - Contenait environ 60 millions de paramètres.
  - **Inception (2014)**:
    - Proposé pour ImageNet.
    - Particularité: Utilise des convolutions de tailles différentes au même niveau de couche pour capturer l'information à diverses échelles.
  - **VGG16 (2015)**:
    - Contenait environ 138 millions de paramètres.
- **Évolution et facteurs clés**:
  - Augmentation constante de la profondeur des réseaux et du nombre de paramètres.
  - Performance accrue du matériel (GPU): plus de puissance de calcul et de mémoire, permettant d'entraîner des modèles de plus en plus complexes et profonds.
- **Architectures plus récentes**:
  - **Transformers (2017)**: "Attention Is All You Need" (papier de recherche). Une architecture particulière qui a marqué une évolution importante.
- **Le principe d'entraînement (ajustement des poids pour corriger les erreurs) reste le même, même si les architectures internes et les "briques" des réseaux deviennent plus complexes.**

## 6. Conclusion Générale

- Le concept d'entraînement des perceptrons, c'est-à-dire ajuster les poids pour corriger les erreurs, demeure la base fondamentale de tous les réseaux de neurones, y compris les plus profonds et complexes actuels.
- L'apprentissage profond applique ce principe pour entraîner des réseaux multicouches capables de résoudre des problèmes complexes et non linéaires.
- Ces techniques sont utilisées dans de nombreuses applications modernes (ex: architectures type Transformers).""",
    "neural-network/2.md": """# Notes on Data Mining - Neural Networks

## Main Takeaway

This document provides an overview of Neural Networks and Deep Learning, covering their history, the fundamental concept of the perceptron and its learning algorithm, and the evolution of deep neural network architectures. The core idea of adjusting weights to correct errors in perceptrons forms the foundation for modern, multi-layer neural networks, which are crucial for solving complex, non-linear problems across various applications.

## 1. Neural Networks

**Source:** Germain Forestier, PhD, Université de Haute-Alsace (https://germain-forestier.info)

### History

- 1960s: Rosenblatt programmed a perceptron (binary classifier).
- 1969: Minsky and Papert highlighted the problem of nonlinear decision boundaries.
- 1970s: AI Winter.
- 1974: Werbos proposed the backpropagation algorithm in his thesis.
- 1986: Rumelhart, Hinton, and Williams rediscovered the backpropagation algorithm.
- 1990s: Convolutional neural networks by Yann LeCun.

**Source:** https://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part1.html

### Perceptron

#### Key Concepts

- Takes n input values (x1, ..., xn), with x0 = 1 (bias).
- Computes an output o.
- Parameters to learn are weights (w0, ..., wn).
- Output o is computed as Σ wi xi.
- This sum is passed through an activation function.
- Activation function: Heaviside function:

  ```
  f(x) = 1 if x > 0
         0 otherwise
  ```

- Perceptrons model linear decision boundaries but cannot solve non-linear problems.

#### Structure Diagram

```
x0 ---- w0 ----
x1 ---- w1 ----
x2 ---- w2 ----
...             
xn ---- wn ---- > Σ > Activation function > Output (o)
```

- Entry weights are learned parameters.

### Perceptron Learning Algorithm

#### "Classic" Algorithm

1. Initialize perceptron weights to arbitrary values.
2. Each time a new example is presented, adjust weights based on correct/incorrect classification.
3. Algorithm stops when all examples are presented without any weight adjustments.

#### Error Correction Algorithm

- **Input:** Set of examples (x1, ..., xn) and expected outputs (c).
- Random initialization of weights wi (for i from 0 to n).
- Repeat:
  - Take an example.
  - Compute perceptron output o for the example.
  - Weight update:
    - For i from 0 to n:
      - wi = wi + (c - o)xi
    - Endfor
  - Endrepeat
- **Output:** Perceptron defined by (w0, w1, ..., wn).
- **Risk:** Oscillation; requires a defined number of epochs.

#### Perceptron Learning Example: Boolean OR

- Initialization: w0 = 0, w1 = 1, w2 = -1.
- Examples Order: (1,0,0,0), (1,0,1,1), (1,1,0,1), (1,1,1,1) (where x0, x1, x2, c).
- The document provides a detailed step-by-step unrolling of the learning process, showing weight adjustments over 10 iterations. The weights adjust based on the error (c - o) until the perceptron converges to correctly classify all OR inputs.

## 2. Deep Learning

### Definition

Deep Learning ≡ Deep Neural Networks ≡ Multi-layer Networks.

### Success

Lies in training multi-layer neural networks to learn nonlinear decisions.

"Deep Learning: a fancy word for deep neural networks."

### Key Deep Neural Network Architectures

#### LeNet-5 (1998): For MNIST-like tasks (document recognition).

**Features:**

- Designed for grayscale images.
- Convolutions followed by pooling, then a final MLP.
- Uses linear functions: sigmoid/tanh.
- Activation functions applied after pooling.
- Same filters applied to each channel.
- Contains 60,000 learnable parameters.

#### AlexNet (2012): For ImageNet.

**Features:**

- Designed for RGB images.
- Similar to LeNet-5 but significantly larger.
- Contains 60 million parameters.
- Uses the ReLU (Rectified Linear Unit) activation function.
- Had the most significant impact on deep learning.

#### Inception (2014): For ImageNet.

**Features:**

- Implements 1x1 convolution to reduce computational cost.
- Uses multiple filter sizes in parallel.
- Eliminates the need to manually choose these hyperparameters.

#### VGG-16 (2015): For ImageNet.

**Features:**

- Same filter size (3x3) for all layers.
- Same max-pooling size (2x2) for all layers.
- Contains 138 million learnable parameters.
- No need to manually choose filters or pooling sizes.

## 3. Conclusion

### Key Points

- The concept of training perceptrons (adjusting weights to correct errors) is the fundamental principle of modern neural networks.
- Deep learning extends this by using multi-layer networks to solve more complex problems.
- The same training approach applies across various applications (e.g., image processing, speech recognition, language models like GPT)."""
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
