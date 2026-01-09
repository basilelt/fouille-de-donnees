## Decision Trees

Advantages:
- Interpretable: Trees are easy to understand and visualize.
- Minimal Data Prep: No need for normalization or scaling.
- Handle Multiple Types: Can deal with numerical and categorical data.
- Non-linear Relationships: Naturally handles non-linearity in data.

Disadvantages:
- Overfitting: Trees can easily become too complex, capturing noise.
- Instability: Small changes in data might result in a completely different tree.
- Bias: Trees are biased to features with more levels.
- Locally Optimal: Greedy algorithms might not always find the optimal tree.

What is a Decision Tree primarily used for?
Classification and regression


What does each internal node of a Decision Tree represent?
A test on an attribute


Which algorithm is widely known for building Decision Trees using information gain?
ID3 (Iterative Dichotomiser 3)


What is 'entropy' in the context of Decision Trees?
A measure of uncertainty


What is overfitting in the context of Decision Trees?
Learning the training data too well, including noise


What does the pruning process in Decision Trees aim to achieve?
Improve generalization by reducing complexity


Which strategy is used to avoid overfitting by penalizing the complexity of the tree?
Cost Complexity Pruning (CCP)


In a Decision Tree, which criterion is commonly used to split nodes?
Information gain


How does a Random Forest help to reduce overfitting?
By averaging predictions of multiple trees


Which type of data can Decision Trees handle effectively?
Both numerical and categorical data


What is the 'Minimum Description Length' principle in pruning Decision Trees?
Prefer the simplest model that fits the data


What does 'entropy gain' represent in Decision Tree construction?
Reduction in uncertainty


Which method is used to handle numerical data in Decision Trees?
Discretization


Which problem arises from allowing a Decision Tree to grow without restriction?
Overfitting


Why are Decision Trees considered interpretable models?
Easy to understand and visualize