# Auto-generated content module for Wooclap
# Contains embedded markdown content for this topic

TOPIC_NAME = "Wooclap"
TOPIC_KEY = "wooclap"

CONTENT = {
    "wooclap/1.md": """# Data Mining & Machine Learning Concepts

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
- Model predicting future value based on weighted sum of past values: Auto Regressive (AR) model.""",
    "wooclap/qcm.md": """# QCM du Cours : Data Mining & Machine Learning

## Intro

* **What is the primary objective of data mining?**
* Discover patterns


* **Which of the following is an example of structured data?**
* A CSV file


* **What is the main purpose of data preprocessing in data mining?**
* Clean data


* **What is the focus of exploratory data analysis (EDA)?**
* Understand data


* **Which method is used for handling missing values by replacing them?**
* Mean imputation


* **What technique is commonly used for reducing data dimensionality?**
* Principal Component Analysis (PCA)


* **Which language is commonly used for managing relational databases?**
* SQL


* **Which of the following is a type of unsupervised learning?**
* Clustering


* **What is the purpose of big data analysis in data mining?**
* Extract insights


* **Which step in the data mining process involves assessing model quality?**
* Evaluation


* **What is the key characteristic of unstructured data?**
* Lacks a specific form


* **Which discipline does data mining heavily intersect with for predictive analytics?**
* Machine learning


* **What does the term "veracity" refer to in the context of big data?**
* Data quality


* **Which data visualization technique is ideal for showing the distribution of a variable?**
* Histogram



## KNN (k-Nearest Neighbors)

* **What type of learning is k-Nearest Neighbors (k-NN) classified under?**
* Instance-based learning


* **Which distance metric is commonly used in the k-NN algorithm?**
* Euclidean Distance


* **What is a primary disadvantage of the k-NN algorithm?**
* High computational cost


* **What is the effect of choosing a small value of 'k' in k-NN?**
* Sensitive to noise


* **Which technique is used to ensure features contribute equally to distance computation in k-NN?**
* Feature scaling


* **Which type of voting in k-NN assigns more influence to closer neighbors?**
* Weighted voting


* **What does 'lazy learning' refer to in the context of k-NN?**
* No model building during training


* **What is the main purpose of one-hot encoding in k-NN?**
* Handle categorical features


* **Which method helps in choosing an optimal value for 'k' in k-NN?**
* Cross-validation


* **What is a real-world application of the k-NN algorithm?**
* Medical diagnosis


* **What is a key challenge of using k-NN with large datasets?**
* High computational complexity


* **In which scenario is weighted voting particularly useful in k-NN?**
* When closer neighbors should have more influence


* **What preprocessing step is critical for k-NN when dealing with features of varying scales?**
* Normalization


* **What is a common strategy to avoid ties in k-NN binary classification problems?**
* Choose an odd value for 'k'



## Decision Trees

* **What is a Decision Tree primarily used for?**
* Classification and regression


* **What does each internal node of a Decision Tree represent?**
* A test on an attribute


* **Which algorithm is widely known for building Decision Trees using information gain?**
* ID3 (Iterative Dichotomiser 3)


* **What is 'entropy' in the context of Decision Trees?**
* A measure of uncertainty


* **What is overfitting in the context of Decision Trees?**
* Learning the training data too well, including noise


* **What does the pruning process in Decision Trees aim to achieve?**
* Improve generalization by reducing complexity


* **Which strategy is used to avoid overfitting by penalizing the complexity of the tree?**
* Cost Complexity Pruning (CCP)


* **In a Decision Tree, which criterion is commonly used to split nodes?**
* Information gain


* **How does a Random Forest help to reduce overfitting?**
* By averaging predictions of multiple trees


* **Which type of data can Decision Trees handle effectively?**
* Both numerical and categorical data


* **What is the 'Minimum Description Length' principle in pruning Decision Trees?**
* Prefer the simplest model that fits the data


* **What does 'entropy gain' represent in Decision Tree construction?**
* Reduction in uncertainty


* **Which method is used to handle numerical data in Decision Trees?**
* Discretization


* **Which problem arises from allowing a Decision Tree to grow without restriction?**
* Overfitting


* **Why are Decision Trees considered interpretable models?**
* Easy to understand and visualize



## Bayes Classifier

* **What is Bayes' Theorem used for?**
* To calculate conditional probabilities


* **What does the Naive Bayesian Classifier assume about attributes?**
* Independence


* **What symbol represents the probability of an event A?**
* 


* **If events A and B are independent, what is ?**
* 


* **What does  represent?**
* The probability of A given B


* **In Bayesian classification, what is the term ?**
* Maximum A Posteriori Hypothesis


* **In the context of text classification, what can an attribute represent?**
* A word in the text


* **What problem occurs if a feature value never appears in the training set?**
* Zero probability


* **What kind of data does the Bayesian Classifier handle well?**
* Small datasets


* **Which scenario is the Bayesian Classifier not well suited for?**
* Complex models with feature interactions


* **What does  represent in Bayesian classification?**
* The prior probability of class 


* **Which method is used to deal with very small probability values in Bayesian classification?**
* Logarithmic transformation


* **In Gaussian Naive Bayes, what is assumed about the distribution of attributes?**
* Gaussian (Normal) distribution


* **Which of the following is a disadvantage of the Naive Bayesian Classifier?**
* Assumes conditional independence of features


* **Which of the following is an example of text classification using Bayesian methods?**
* Spam email filtering



## Hierarchical Clustering

* **What is the primary goal of clustering?**
* Divide a set of objects into groups


* **What type of task is clustering considered?**
* Unsupervised


* **Which distance measure is commonly used for numerical values in clustering?**
* Euclidean distance


* **What is the purpose of scaling attributes in clustering?**
* To ensure each attribute contributes equally


* **What does a dendrogram represent in hierarchical clustering?**
* Hierarchical representation of successive merges


* **Which linkage method is sensitive to noise in hierarchical clustering?**
* Single linkage


* **What is the main advantage of Ward’s linkage method?**
* Creates balanced clusters


* **How do you determine the number of clusters in hierarchical clustering?**
* Cut the tree at a specific height


* **What type of evaluation criterion is the Silhouette Coefficient?**
* Internal


* **What does the Adjusted Rand Index compare?**
* Clustering results to true labels


* **What is a major disadvantage of hierarchical clustering?**
* Computational cost of the distance matrix


* **In hierarchical clustering, how are clusters formed initially?**
* Each object is its own cluster


* **What is the centroid linkage method known for?**
* Good resistance to noise


* **What happens to the number of clusters during the hierarchical clustering process?**
* Decreases over time


* **Which criterion is based on inertia in hierarchical clustering?**
* Ward's linkage



## KMeans

* **What is the primary objective of K-Means clustering?**
* Minimize intra-cluster variance


* **Which step comes first in the K-Means algorithm?**
* Choose k random points as cluster centers


* **How are cluster centers (centroids) updated in K-Means?**
* By calculating the mean of the points in each cluster


* **What is K in K-Means?**
* Number of clusters


* **What technique is used to initialize centroids to improve convergence in K-Means?**
* K-Means++


* **What does convergence in K-Means mean?**
* When centroids no longer change


* **Which distance metric is commonly used in K-Means?**
* Euclidean distance


* **What issue arises due to outliers in K-Means?**
* It distorts centroids


* **Which method helps to decide the number of clusters (k) in K-Means?**
* Elbow Method


* **What is a common limitation of K-Means?**
* Assumes spherical clusters


* **Which of the following is a variant of K-Means?**
* K-Medoids


* **What type of learning is K-Means clustering?**
* Unsupervised learning


* **How does the Silhouette Score help in K-Means clustering?**
* By evaluating the quality of clustering


* **What happens if k is too large in K-Means?**
* Clusters may overlap too much


* **What is the main reason to use K-Means++?**
* To improve initial centroid selection



## Neural Networks

* **Who developed the perceptron in the 1960s?**
* Rosenblatt


* **What is the main limitation of perceptrons?**
* Cannot solve non-linear problems


* **Who proposed the backpropagation algorithm in 1974?**
* Werbos


* **What activation function was initially used in perceptrons?**
* Heaviside


* **What is a key feature of LeNet-5?**
* Designed for grayscale images


* **What distinguishes AlexNet from LeNet-5?**
* Handles RGB images


* **Which model introduced the concept of parallel filter sizes?**
* Inception


* **How many learnable parameters does VGG-16 have?**
* 138 million


* **What principle is foundational to neural network training?**
* Error correction


* **What technique does Inception use to reduce computational cost?**
* 1×1 convolutions


* **What activation function is commonly used in AlexNet?**
* ReLU


* **What was a major achievement of AlexNet in 2012?**
* Popularized deep learning


* **What is the primary advantage of deep neural networks?**
* Non-linear decision boundaries


* **Who highlighted the problem of non-linear decision boundaries?**
* Minsky and Papert


* **What is the output of a perceptron passed through?**
* Activation function



## Time Series

* **What is the key property of time series data?**
* Temporal order matters


* **What does Z-Normalization do to a time series?**
* Standardizes the series to mean 0 and standard deviation 1


* **What is the primary goal of time series forecasting?**
* Predict future values based on past data


* **Which technique aligns time series of unequal lengths?**
* Dynamic Time Warping (DTW)


* **What component of time series represents long-term progression or direction?**
* Trend


* **Which imputation method fills missing values by using the last observed value?**
* Forward Fill


* **What is the purpose of the Fourier Transform in time series?**
* Decomposes the series into frequency components


* **Which method is best for capturing cyclic patterns in time series?**
* Fourier Transform


* **Which method is used for filling missing time series data by replacing each missing value with the median of the observed values?**
* Forward Fill


* **What is the key advantage of Dynamic Time Warping (DTW) in time series classification?**
* It handles temporal distortions


* **What statistical feature captures the spread or dispersion of a time series?**
* Variance


* **Which model is often used to predict the future value based on the weighted sum of past values in time series analysis?**
* Auto Regressive (AR) model"""
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
