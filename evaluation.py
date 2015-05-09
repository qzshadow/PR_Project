# -*- coding: utf-8 -*-
__author__ = 'Qin'

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from import_iris_data import generate_data
from ANN_model import MLP_1HL

# Generate classification objects.
nn = MLP_1HL()
rfc = RandomForestClassifier(n_estimators=50)

X,labels,n_folds,skf = generate_data()

# Generate arrays for meta-level training and testing sets, which are n x len(clfs).
scores_nn = np.zeros(n_folds) # scores for nn
scores_rfc = np.zeros(n_folds) # scores for rfc

print('Training classifiers...')
# Iterate over the folds, each with training set and validation set indicies.
for i, (train_index, test_index) in enumerate(skf):
    print('  Fold {0}'.format(i))

    # Generate the training set for the fold.
    X_train = X[train_index]
    y_train = labels[train_index]

    # Generate the testing set for the fold.
    X_test = X[test_index]
    y_test = labels[test_index]

    # Train the models on the training set.
    # We time the training using the built-in timeit magic function.
    print('    Neural Network: ',
    nn.fit(X_train, y_train))
    print('    Random Forest: ',
    rfc.fit(X_train, y_train))

    # Evaluate the models on the testing set.
    scores_nn[i] = metrics.accuracy_score(y_test, nn.predict(X_test))
    scores_rfc[i] = metrics.accuracy_score(y_test, rfc.predict(X_test))
print('Done training classifiers.')
print("\n")

# The mean of the scores on the testing set.
print('Artificial Neural Network Accuracy = %s' % (scores_nn.mean(axis=0)))
print('Random Forest Accuracy = %s' % (scores_rfc.mean(axis=0)))