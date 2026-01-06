# Data Mining & Machine Learning Concepts

These notes provide a concise overview of fundamental concepts in data mining, machine learning algorithms (K-NN, Decision Trees, Bayes Classifier, Neural Networks), and clustering techniques (Hierarchical Clustering, K-Means), along with an introduction to Time Series analysis. They cover primary objectives, data types, preprocessing steps, algorithm characteristics, advantages, disadvantages, and key terminology.

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
- Method filling missing values with median of observed values: Forward Fill. (Note: This seems like a repeat or slight variation of "Forward Fill" answer, but input text states "median of observed values" for a method identified as Forward Fill.)
- Key advantage of DTW in classification: It handles temporal distortions.
- Statistical feature capturing spread: Variance.
- Model predicting future value based on weighted sum of past values: Auto Regressive (AR) model.