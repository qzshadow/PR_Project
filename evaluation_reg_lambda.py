# -*- coding: utf-8 -*-
__author__ = 'Qin'

from sklearn import metrics
import numpy as np
import random
from import_iris_data import generate_data
from ANN_model import MLP_1HL
from matplotlib import pyplot as plt

# Generate classification objects.

X,labels,n_folds,skf = generate_data(label_encode=True,n_folds=10,iris_path='./iris.data')
# Generate arrays for meta-level training and testing sets, which are n x len(clfs).
scores_nn = np.zeros(n_folds) # scores for nn
result = np.zeros(101)
for idx, reg_lambda in enumerate(np.linspace(0,1,101)):
    nn = MLP_1HL(reg_lambda=reg_lambda,epsilon_init=0.2,hidden_layer_size=25,opti_method='TNC',maxiter=500,load_theta0=True,activation_func='sigmoid')
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

        # Evaluate the models on the testing set.
        scores_nn[i] = metrics.accuracy_score(y_test, nn.predict(X_test))
    print('Done training classifiers.')
    print("\n")

    # The mean of the scores on the testing set.
    print('Artificial Neural Network Accuracy = %s' % (scores_nn.mean(axis=0)))
    result[idx] = scores_nn.mean(axis=0)
    
plt.plot(np.linspace(0,1,101),result,label='accuracy',color='red')
plt.xlabel('reg_lambda')
plt.ylabel('accuracy')
plt.title('accuracy plot by changing the reg_lambda')
plt.legend(loc=4)
plt.show()

def myShuffle(X,y):
    num_features = len(X[0])
    sample = np.hstack((X,y))
    random.shuffle(sample)
    return sample[:,:num_features],sample[:,num_features:]