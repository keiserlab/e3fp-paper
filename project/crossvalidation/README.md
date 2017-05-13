# K-fold Cross-Validation

Here we assess the performance of E3FP relative to ECFP4, a widely used 2D
fingerprint. We do so using *k*-fold cross-validation on ChEMBL20 using
various classifiers. See directories for instructions specific to SEA or other
classifiers:

- [`sea`](sea): *k*-fold cross-validation of various flavors of ECFP4 and E3FP
  using the Similarity Ensemble Approach (SEA) as classifier.
- [`classifiers`](classifiers): validation using four classifiers: Naive Bayes (NB), Random Forests (RF), Support Vector Machine with a linear kernel (LinSVM), and Artificial Neural Network (NN).
- [`classifiers_mean`](classifiers_mean): validation using the four classifiers listed above, but combining all bit fingerprints for a molecule into a single 'float' fingerprint by averaging.
