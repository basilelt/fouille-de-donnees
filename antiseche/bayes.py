## Bayes Classifier

Advantages:
- Simplicity: Easy to implement and computationally efficient.
- Interpretability: Produces probabilities that are easy to interpret and understand.
- Works Well with Small Data: Performs well even with a small dataset, especially when features are independent.
- Scalability: Can handle a large number of features, making it suitable for text classification and spam filtering.

Disadvantages:
- Conditional Independence Assumption: Assumes that all features are independent given the class, which is often not true in real-world data.
- Zero Probabilities: If a particular feature value never occurs in the training data, it can lead to zero probabilities unless smoothing is applied.
- Not Suitable for Complex Models: May not perform well on more complex tasks where feature interactions are important.


What is Bayes' Theorem used for?
To calculate conditional probabilities


What does the Naive Bayesian Classifier assume about attributes?
Independence


What symbol represents the probability of an event A?
P(A)


If events A and B are independent, what is P(A inter B) ?
P(A)xP(B)


What does P(A|B) represent?
The probability of A given B


In Bayesian classification, what is the term hmap?
Maximum A Posteriori Hypothesis


In the context of text classification, what can an attribute represent?
A word in the text


What problem occurs if a feature value never appears in the training set?
Zero probability


What kind of data does the Bayesian Classifier handle well?
Small datasets


Which scenario is the Bayesian Classifier not well suited for?
Complex models with feature interactions


What does P(ck) represent in Bayesian classification?
The prior probability of class ck


Which method is used to deal with very small probability values in Bayesian classification?
Logarithmic transformation


In Gaussian Naive Bayes, what is assumed about the distribution of attributes?
Gaussian (Normal) distribution


Which of the following is a disadvantage of the Naive Bayesian Classifier?
Assumes conditional independence of features


Which of the following is an example of text classification using Bayesian methods?
Spam email filtering