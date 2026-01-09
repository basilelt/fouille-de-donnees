## KMeans

Advantages:
- Efficiency: Operates in linear time, making it very fast.
- Interpretability: Clusters are easy to understand, centered around clear prototypes.
- Simplicity: Minimal parameters required—just the number of clusters and iterations.

Disadvantages:
- Fixed k: Number of clusters must be specified in advance.
- Prototypical Constraints: Limited to finding spherical-shaped clusters around centroids.
- Initialization Sensitivity: Outcome heavily depends on the initial position of centroids.


What is the primary objective of K-Means clustering?
Minimize intra-cluster variance


Which step comes first in the K-Means algorithm?
Choose k random points as cluster centers


How are cluster centers (centroids) updated in K-Means?
By calculating the mean of the points in each cluster


What is K in K-Means?
Number of clusters


What technique is used to initialize centroids to improve convergence in K-Means?
K-Means++


What does convergence in K-Means mean?
When centroids no longer change


Which distance metric is commonly used in K-Means?
Euclidean distance


What issue arises due to outliers in K-Means?
It distorts centroids


Which method helps to decide the number of clusters (k) in K-Means?
Elbow Method


What is a common limitation of K-Means?
Assumes spherical clusters


Which of the following is a variant of K-Means?
K-Medoids


What type of learning is K-Means clustering?
Unsupervised learning


How does the Silhouette Score help in K-Means clustering?
By evaluating the quality of clustering


What happens if k is too large in K-Means?
Clusters may overlap too much


What is the main reason to use K-Means++?
To improve initial centroid selection