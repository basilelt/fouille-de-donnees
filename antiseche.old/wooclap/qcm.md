# QCM du Cours : Data Mining & Machine Learning

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
* Auto Regressive (AR) model